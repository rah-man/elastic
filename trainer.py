import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models.feature_extraction import create_feature_extractor

from base import BaseDataset, Extractor, get_data
from model import SingleMLP
from regulariser import LwF, Regulariser
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
        regulariser=None,
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
        self.regulariser = None

        self.seen_cls = 0
        self.model = None
        self.previous_model = None
        self.cls2idx = {v: k for k, v in enumerate(self.class_order)}
        self.idx2cls = {k: v for k, v in enumerate(self.class_order)}

        if regulariser == "lwf":
            # initialise LwF with empty model
            self.regulariser = LwF(device=device)

        self.init_model()

    def init_model(self):
        if not self.regulariser:
            self.model = SingleMLP(out_features=self.total_cls).to(self.device)
        # with regulariser, e.g. LwF, the model is updated for every iteration (?)
        # is there another approach where the model output units already predefined ?

    def train_loop(self):        
        for task in range(len(self.dataset)):  

            # update the model for new classes in each iteration for LwF        
            new_cls = len(self.dataset[task]["classes"])
            if isinstance(self.regulariser, Regulariser):
                if self.model:
                    self.previous_model = copy.deepcopy(self.model)
                    self.previous_model.eval()
                
                self.model = self.regulariser.update_model(self.seen_cls, new_cls)

            print(f"TRAINING TASK-{task}\tCLASSES: {self.dataset[task]['classes']}")

            trainloader, valloader = self._get_current_dataloader(task) 
            self.model = self.model.to(self.device)           
            self.optimiser = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=WEIGHT_DECAY)

            for epoch in range(self.epochs):
                train_loss, dist_loss, val_loss = [], [], []
                self.model.train()

                print(f"Epoch: {epoch+1}/{self.epochs}")

                for i, (x, y) in enumerate(trainloader):
                    x = x.to(self.device)
                    y = y.to(self.device)

                    # outputs = self.model(self.extractor(x))
                    outputs = self.model(x)
                    loss = self.criterion(outputs, y)

                    if self.previous_model:
                        with torch.no_grad():
                            # prev_dist = self.previous_model(self.extractor(x))
                            prev_dist = self.previous_model(x)
                        distillation_loss = self.regulariser.calculate_loss(outputs, prev_dist)
                        loss = self.regulariser.lambda_ * distillation_loss + loss     
                        dist_loss.append(distillation_loss.item())   
     
                    loss.backward()
                    self.optimiser.step()
                    self.optimiser.zero_grad()

                    train_loss.append(loss.item())
                    if (i+1) % self.print_every == 0:
                        message = f"\tStep {i+1}/{len(trainloader)}\tTrain Loss: {np.average(train_loss):.4f}"
                        if dist_loss:
                            message += f" \tDistillation Loss: {np.average(dist_loss):.4f}"
                        print(message)

                if valloader:
                    self.model.eval()
                    for x, y in valloader:
                        x = x.to(self.device)
                        y = y.to(self.device)

                        with torch.no_grad():
                            # outputs = self.model(self.extractor(x))
                            outputs = self.model(x)
                            loss = self.criterion(outputs, y)
                        val_loss.append(loss.item())
                    print(f"\t\t\tVal Loss: {np.average(val_loss):.4f}")
            self.test_loop(task)
            print()

            self.seen_cls += new_cls

    def test_loop(self, current_task=None):
        self.model.to(self.device)
        self.model.eval()
        print("CALCULATING TEST LOSS PER TASK")

        num_test = len(self.dataset) if current_task is None else current_task + 1
        for task in range(num_test):
            test_loss = []
            x_test, y_test = self.dataset[task]["test"]["x"], self.dataset[task]["test"]["y"]
            testloader = self._get_dataloader(x_test, y_test)
            
            for x, y in testloader:
                x = x.to(self.device)
                y = y.to(self.device)

                with torch.no_grad():
                    # outputs = self.model(self.extractor(x))
                    outputs = self.model(x)
                    loss = self.criterion(outputs, y)
                test_loss.append(loss.item())

            print(f"\tTASK-{task}\tCLASSES: {self.dataset[task]['classes']}\tLoss: {np.average(test_loss):.4f}")


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
        trainloader = self._get_dataloader(x_train, y_train, shuffle=True)

        valloader = None
        if len(current_dataset["val"]["x"]) != 0:
            x_val, y_val = current_dataset["val"]["x"], current_dataset["val"]["y"]
            valloader = self._get_dataloader(x_val, y_val)

        return trainloader, valloader
    
    def _get_dataloader(self, x, y, shuffle=False):
        if self.regulariser:
            _dataset = BaseDataset(x, y, self.transform, self.cls2idx)
        else:
            _dataset = BaseDataset(x, y, self.transform)
        _dataloader = DataLoader(_dataset, batch_size=self.batch_size, shuffle=shuffle)
        return _dataloader


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
        validation=0.2,)

    # not needed anymore
    # extractor = Extractor(base_extractor, weights=weights, return_nodes=return_nodes, device=device)

    trainer = Trainer(criterion, data, lr, n_class, class_order, batch_size=16, epochs=2, device=device, replay=None, regulariser=lwf)

    trainer.train_loop()
    trainer.test_loop()

