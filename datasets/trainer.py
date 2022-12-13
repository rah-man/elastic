import copy
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models.feature_extraction import create_feature_extractor

from base import BaseDataset, Extractor, get_data
from model import IncrementalModel, MultiHeadModel
from strategy import LwF, Strategy
from replay import RandomReplay


WEIGHT_DECAY = 5e-4 # from LwF paper

class Trainer:
    def __init__(
        self, 
        criterion, 
        dataset,
        lr,
        total_cls,
        class_order,
        extractor=None,
        batch_size=64,
        epochs=5,
        device="cpu",        
        replay=None,
        strategy=None,
        transform=None,
        print_every=50,):
    
        self.criterion = criterion
        self.dataset = dataset
        self.extractor = extractor
        self.lr = lr        
        self.total_cls = total_cls
        self.class_order = class_order
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.replay = replay
        self.transform = transform
        self.print_every = print_every
        self.strategy = None

        self.seen_cls = 0
        self.model = None
        self.previous_model = None
        self.cls2idx = {v: k for k, v in enumerate(self.class_order)}
        self.idx2cls = {k: v for k, v in enumerate(self.class_order)}

        if strategy == "lwf":
            self.strategy = LwF()

        # self.model = MultiHeadModel()
        self.model = IncrementalModel()

    def train(self):      
        # TASK ITERATOR  
        for task in range(len(self.dataset)):  

            # update the model for new classes in each iteration    
            cls_labels = self.dataset[task]["classes"]
            new_cls = len(cls_labels)
            self.previous_model = copy.deepcopy(self.model) if self.model.not_none() else None
            self.model.update_model(self.seen_cls, new_cls, cls_labels)
            self.model = self.model.to(device)
            self.optimiser = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=WEIGHT_DECAY)

            print(f"TRAINING TASK-{task}\tCLASSES: {self.dataset[task]['classes']}")

            trainloader, valloader = self._get_current_dataloader(task) 

            # EPOCH ITERATOR
            for epoch in range(self.epochs):
                train_loss, dist_loss, val_loss = [], [], []
                self.model.train()

                print(f"Epoch: {epoch+1}/{self.epochs}")

                # run training on trainloader
                train_loss, dist_loss = self.train_loop(task, trainloader)

            if valloader:
                self.model.eval()
                self.eval_loop(task, valloader)                
            
            # self.test_loop(task)
            print()

            self.seen_cls += new_cls

    def train_loop(self, task, loader):
        train_loss, dist_loss = [], []
        for i, (x, y) in enumerate(loader):
            x = x.to(self.device)
            y = y.to(self.device)

            # outputs = self.model(self.extractor(x))
            if isinstance(self.model, MultiHeadModel):
                outputs = self.model(x)[task]
            else:
                outputs = self.model(x)
            loss = self.criterion(outputs, y)

            if self.previous_model and isinstance(self.strategy, LwF):
                self.previous_model.eval()
                with torch.no_grad():
                    # prev_dist = self.previous_model(self.extractor(x))
                    prev_dist = self.previous_model(x)
                distillation_loss = self.strategy.calculate_loss(outputs, prev_dist, self.seen_cls)
                loss = self.strategy.lambda_ * distillation_loss + loss     
                dist_loss.append(distillation_loss.item())   

            loss.backward()
            self.optimiser.step()
            self.optimiser.zero_grad()

            train_loss.append(loss.item())
            if (i+1) % self.print_every == 0:
                message = f"\tStep {i+1}/{len(loader)}\tTrain Loss: {np.average(train_loss):.4f}"
                if dist_loss:
                    message += f" \tDistillation Loss: {np.average(dist_loss):.4f}"
                print(message)

        return train_loss, dist_loss

    def eval_loop(self, task, loader):
        val_loss, y_true, y_pred = [], [], []
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)

            with torch.no_grad():
                # outputs = self.model(self.extractor(x))
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
            
            val_loss.append(loss.item())

            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(outputs.cpu())
        self._calculate_metric(task, y_true, y_pred)
        # print(f"\t\t\tEval Loss: {np.average(val_loss):.4f}")

    def test_loop(self, current_task=None):
        self.model.to(self.device)
        self.model.eval()
        print("CALCULATING TEST ACCURACY PER TASK")

        num_test = len(self.dataset) if current_task is None else current_task + 1
        for task in range(num_test):
            # test_loss = []
            micro, macro = [], []
            x_test, y_test = self.dataset[task]["test"]["x"], self.dataset[task]["test"]["y"]
            testloader = self._get_dataloader(x_test, y_test, shuffle=False)
            
            for x, y in testloader:
                x = x.to(self.device)
                y = y.to(self.device)

                with torch.no_grad():
                    # outputs = self.model(self.extractor(x))
                    outputs = self.model(x)
                    # loss = self.criterion(outputs, y)
                # test_loss.append(loss.item())

            ytrue, ypred = self._map2class(y.cpu().numpy().tolist(), outputs.cpu())
            micro = f1_score(ytrue, ypred, average="micro") * 100
            macro = f1_score(ytrue, ypred, average="macro") * 100
            # print(f"\tTASK-{task}\tCLASSES: {self.dataset[task]['classes']}\tLoss: {np.average(test_loss):.4f}")
            print(f"\tTask-{task}\tClasses: {self.dataset[task]['classes']}\tf1-micro: {micro:.2f}\tf1-macro: {macro:.2f}")

    def _get_current_dataloader(self, task):
        current_dataset = self.dataset[task]
        if self.replay:
            # update the buffer from the previous tasks' datasets
            # and append current task's dataset at the end
            self.replay.update_buffer(current_dataset)
            x_train, y_train = self.replay.buffer["x"], self.replay.buffer["y"]
        else:
            # if not replay, just use current task's dataset
            x_train, y_train = current_dataset["train"]["x"], current_dataset["train"]["y"]

        if isinstance(self.model, IncrementalModel):
            cls2idx = self.cls2idx
        elif isinstance(self.model, MultiHeadModel):
            cls2idx = self.model.cls2idx[-1]
        trainloader = self._get_dataloader(x_train, y_train, cls2idx)

        valloader = None
        if len(current_dataset["val"]["x"]) != 0:
            x_val, y_val = [], []
            for i in range(task+1):
                x_val.extend(self.dataset[i]["val"]["x"])
                y_val.extend(self.dataset[i]["val"]["y"])
                # x_val, y_val = current_dataset["val"]["x"], current_dataset["val"]["y"]
            valloader = self._get_dataloader(x_val, y_val, cls2idx, shuffle=False)

        return trainloader, valloader
    
    def _get_dataloader(self, x, y, cls2idx, shuffle=True):
        # if self.strategy:
        _dataset = BaseDataset(x, y, self.transform, cls2idx)
        # else:
        #     _dataset = BaseDataset(x, y, self.transform)
        _dataloader = DataLoader(_dataset, batch_size=self.batch_size, shuffle=shuffle)
        return _dataloader

    def _map2class(self, ytrue, ypred):        
        ypred = [torch.argmax(y).item() for y in ypred]
        ytrue = list(map(lambda x: self.idx2cls[x], ytrue))
        ypred = list(map(lambda x: self.idx2cls[x], ypred))

        return ytrue, ypred

    def _calculate_metric(self, task, ytrue, ypred):
        base_classes, old_classes, new_classes = [], [], []
        micro, macro = [], []
        ytrue, ypred = self._map2class(ytrue, ypred)

        base_classes = self.dataset[0]["classes"]
        for i in range(task):
            old_classes.extend(self.dataset[i]["classes"])
        new_classes = self.dataset[task]["classes"]

        base_indices, old_indices, new_indices = [[
            idx for idx, val in enumerate(ytrue) if val in cls] 
            for cls in [base_classes, old_classes, new_classes]]

        # print(f"base_classes: {base_classes}\tytrue: {np.unique(ytrue)}\tbase_indices: {base_indices}")
        # print(f"old_classes: {old_classes}\tytrue: {np.unique(ytrue)}\told_indices: {old_indices}")
        # print(f"new_classes: {new_classes}\tytrue: {np.unique(ytrue)}\tnew_indices: {new_indices}")

        for indices in [base_indices, old_indices, new_indices]:
            filtered = [(pred_val, true_val) for idx, (pred_val, true_val) in enumerate(zip(ypred, ytrue)) if idx in indices]
            _y_pred = [each[0] for each in filtered]
            _y_true = [each[1] for each in filtered]
            micro.append(f1_score(_y_true, _y_pred, average="micro") * 100 if len(indices) > 0 else micro[-1])
            macro.append(f1_score(_y_true, _y_pred, average="macro") * 100 if len(indices) > 0 else macro[-1])

        micro.append(f1_score(ytrue, ypred, average="micro") * 100)
        macro.append(f1_score(ytrue, ypred, average="macro") * 100)

        print("f1-micro")
        print(f"\tBase: {micro[0]:.2f}")
        print(f"\tOld: {micro[1]:.2f}")
        print(f"\tNew: {micro[2]:.2f}")
        print(f"\tAll: {micro[3]:.2f}")

        print("f1-macro")
        print(f"\tBase: {macro[0]:.2f}")
        print(f"\tOld: {macro[1]:.2f}")
        print(f"\tNew: {macro[2]:.2f}")
        print(f"\tAll: {macro[3]:.2f}")        
        print()


if __name__ == "__main__":
    lr = 0.001
    n_class = 10
    criterion = nn.CrossEntropyLoss()

    # the following dataset and embedding pointers are not needed anymore
    # dataset = datasets.CIFAR10
    # data_path = "CIFAR_data/"
    # base_extractor = models.vit_b_16
    # weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
    # transform = weights.transforms()
    # return_nodes = ["getitem_5"]

    train_embedding_path = "cifar10_train_embedding.pt"
    test_embedding_path = "cifar10_test_embedding.pt"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random_replay = RandomReplay(mem_size=3000)
    lwf = "lwf"

    data, class_order = get_data(
        train_embedding_path, 
        test_embedding_path, 
        num_tasks=5, 
        validation=0.2,
        seed=42)
    # print("class_order", class_order)
    # not needed anymore
    # extractor = Extractor(base_extractor, weights=weights, return_nodes=return_nodes, device=device)

    trainer = Trainer(criterion, data, lr, n_class, class_order, batch_size=32, epochs=3, device=device, replay=None, strategy=None)

    start_time = time.time_ns()
    trainer.train()
    end_time = time.time_ns()
    print(f"Total elapsed time: {(end_time - start_time)/1e9:.4f} seconds")

    # trainer.test()

