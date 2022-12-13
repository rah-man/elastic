import argparse
import copy
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from base import BaseDataset, get_data
from mets import Metrics
from model import SingleMLP
from sklearn.metrics import accuracy_score

WEIGHT_DECAY = 5e-4 # from LwF paper

class Trainer:
    def __init__(
        self, 
        criterion, 
        dataset,
        lr,
        wd,
        mom,
        total_cls,
        batch_size=64,
        epochs=5,
        metric=None,):
    
        self.criterion = criterion
        self.dataset = dataset
        self.lr = lr        
        self.wd = wd
        self.mom = mom
        self.total_cls = total_cls
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = metric
    
        self.seen_cls = 0
        self.cls2idx = {}
        self.idx2cls = {}
        self.train_loss = []
        self.dist_loss = []
        self.val_loss = []
        self.model = None
        self.previous_model = None
        self.lambda_= 1.6
        self.temperature = 2

    def update_classmap(self, new_cls):
        cls_ = list(self.cls2idx.keys())
        cls_.extend(new_cls)
        self.cls2idx = {v: k for k, v in enumerate(cls_)}
        self.idx2cls = {k: v for k, v in enumerate(cls_)}

    def _update_model(self, seen_cls, new_cls):
        self.new_cls = new_cls

        if not self.model:
            model = SingleMLP(out_features=new_cls, bias=False)
            self._kaiming_normal_init(model.fc)
        else:
            weights = self.model.fc.weight
            out_features = seen_cls + new_cls
            
            model = SingleMLP(out_features=out_features, bias=False)
            self._kaiming_normal_init(model.fc)
            model.fc.weight.data[:seen_cls] = weights
            
        self.model = model
        return self.model  

    def _kaiming_normal_init(self, model):
        if isinstance(model, nn.Conv2d):
            nn.init.kaiming_normal_(model.weight, nonlinearity="relu")
        if isinstance(model, nn.Linear):
            nn.init.kaiming_normal_(model.weight, nonlinearity="sigmoid")

    def calculate_loss(self, current_dist, prev_dist):
        # take the current model outputs for old classes
        logits_dist = current_dist[:, :self.seen_cls]
        log_p = torch.log_softmax(logits_dist / self.temperature, dim=1)
        q = torch.softmax(prev_dist / self.temperature, dim=1)
        distil_loss = nn.functional.kl_div(log_p, q, reduction="batchmean")
        return distil_loss

    def train_loop(self, steps=2):
        val_loaders = []        
        
        for task in range(len(self.dataset)):  
            print(f"TRAINING TASK-{task}\tCLASSES: {self.dataset[task]['classes']}")

            new_cls = len(self.dataset[task]["classes"])
            self.update_classmap(self.dataset[task]["classes"])        

            if self.model:
                self.previous_model = copy.deepcopy(self.model)
                self.previous_model.eval()

            self.model = self._update_model(self.seen_cls, new_cls).to(self.device)
            optimiser = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.wd, momentum=self.mom)
            scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=30)
            
            # for logging
            print(optimiser)
            print()

            trainloader, valloader = self._get_current_dataloader(task) 
            val_loaders.append(valloader)

            tloss, dloss = [], []
            for epoch in range(self.epochs):
                ypreds, ytrue = [], []
                self.model.train()

                running_train_loss, running_dist_loss = 0.0, 0.0
                dataset_len = 0

                for i, (x, y) in enumerate(trainloader):
                    x = x.to(self.device)
                    y = y.to(self.device)

                    outputs = self.model(x)
                    loss = self.criterion(outputs, y)

                    if self.previous_model:
                        with torch.no_grad():
                            prev_dist = self.previous_model(x)
                        distillation_loss = self.calculate_loss(outputs, prev_dist)
                        loss = self.lambda_ * distillation_loss + loss                     
                        running_dist_loss += distillation_loss * x.size(0)

                    loss.backward()
                    optimiser.step()
                    scheduler.step()
                    optimiser.zero_grad()                    
                                  
                    predicted = torch.argmax(outputs.data, 1)
                    ypreds.extend(predicted.detach().cpu().tolist())
                    ytrue.extend(y.detach().cpu().tolist())

                    running_train_loss += loss.item() * x.size(0)
                    dataset_len += x.size(0)
                
                tloss.append(running_train_loss / dataset_len)            
                dloss.append(running_dist_loss / dataset_len)
                self.train_loss.append(tloss)
                self.dist_loss.append(dloss)
                print(f"Epoch: {epoch+1}/{self.epochs}\ttrain_loss: {tloss[-1]:.4f}\ttrain_accuracy: {(100 * accuracy_score(ytrue, ypreds)):.4f}")
            
            if val_loaders:
                self.model.eval()                                
                vloss = []
                for i, valloader in enumerate(val_loaders):
                    running_val_loss = 0.0
                    dataset_len = 0
                    ypreds, ytrue = [], []
                    for x, y in valloader:
                        x = x.to(self.device)
                        y = y.to(self.device)

                        with torch.no_grad():
                            outputs = self.model(x)
                            loss = self.criterion(outputs, y)
                            running_val_loss += loss.item() * x.size(0)
                        
                        predicted = torch.argmax(outputs.data, 1)
                        ypreds.extend(predicted.detach().cpu().tolist())
                        ytrue.extend(y.detach().cpu().tolist())
                        dataset_len += x.size(0)
                    
                    vloss.append(running_val_loss / dataset_len)                    
                    task_accuracy = 100 * accuracy_score(ytrue, ypreds)
                    print(f"\tTask-{i} val_loss: {vloss[-1]:.4f}\tval_accuracy: {task_accuracy:.4f}")
                    if self.metric:
                        self.metric.add_accuracy(task, task_accuracy)
                self.val_loss.append(vloss)
            
            if self.metric:
                self.metric.add_forgetting(task)
            print()
            self.seen_cls += new_cls

        return self.train_loss, self.val_loss

    def test_loop(self, current_task=None):
        self.model.to(self.device)
        self.model.eval()
        print("CALCULATING TEST LOSS PER TASK")

        num_test = len(self.dataset) if current_task is None else current_task + 1     
        all_preds, all_true = [], []   
        for task in range(num_test):
            ypreds, ytrue = [], []
            x_test, y_test = self.dataset[task]["test"]["x"], self.dataset[task]["test"]["y"]
            testloader = self._get_dataloader(x_test, y_test)
            
            for x, y in testloader:
                x = x.to(self.device)
                y = y.to(self.device)

                with torch.no_grad():
                    outputs = self.model(x)                    
                predicted = torch.argmax(outputs.data, 1)
                ypreds.extend(predicted.detach().cpu().tolist())
                ytrue.extend(y.detach().cpu().tolist())
                
                all_preds.extend(predicted.detach().cpu().tolist())
                all_true.extend(y.detach().cpu().tolist())

            print(f"\tTASK-{task}\tCLASSES: {self.dataset[task]['classes']}\ttest_accuracy: {(100 * accuracy_score(ytrue, ypreds)):.4f}")
        print(f"All accuracy: {(100 * accuracy_score(all_true, all_preds)):.4f}")

    def _get_current_dataloader(self, task):
        current_dataset = self.dataset[task]        
        x_train, y_train = current_dataset["train"]["x"], current_dataset["train"]["y"]
        trainloader = self._get_dataloader(x_train, y_train, shuffle=True)

        valloader = None
        if len(current_dataset["val"]["x"]) != 0:
            x_val, y_val = current_dataset["val"]["x"], current_dataset["val"]["y"]
            valloader = self._get_dataloader(x_val, y_val)

        return trainloader, valloader
    
    def _get_dataloader(self, x, y, shuffle=False):
        _dataset = BaseDataset(x, y, None, self.cls2idx)
        _dataloader = DataLoader(_dataset, batch_size=self.batch_size, shuffle=shuffle)
        return _dataloader


if __name__ == "__main__":
    train_embedding_path = "cifar100_train_embedding.pt"
    test_embedding_path = "cifar100_test_embedding.pt"
    val_embedding_path = None

    # train_embedding_path = "imagenet1000_train_embedding.pt"
    # test_embedding_path = None
    # val_embedding_path = "imagenet1000_val_embedding.pt"

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task")
    parser.add_argument("-f", "--file")
    parser.add_argument("-lr", "--learning_rate")
    parser.add_argument("-wd", "--weight_decay")
    parser.add_argument("-mom", "--momentum")    
    args = parser.parse_args()

    n_task = int(args.task)
    f = args.file
    lr = float(args.learning_rate)
    wd = float(args.weight_decay)
    mom = float(args.momentum)
    batch = 256
    n_class = 100
    epochs = 100
    criterion = nn.CrossEntropyLoss()

    data, class_order = get_data(
        train_embedding_path, 
        test_embedding_path, 
        val_embedding_path=val_embedding_path,
        num_tasks=n_task, # num_tasks == n_experts
        validation=0.2,
    )

    met = Metrics()
    trainer = Trainer(
        criterion, data, lr, 
        wd, mom, n_class,
        batch_size=batch, 
        epochs=epochs,
        metric=met,)

    walltime_start, processtime_start = time.time(), time.process_time()
    train_loss, val_loss = trainer.train_loop()
    walltime_end, processtime_end = time.time(), time.process_time()
    elapsed_walltime = walltime_end - walltime_start
    elapsed_processtime = processtime_end - processtime_start
    print('Execution time:', )
    print(f"CPU time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_processtime))}\tWall time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_walltime))}")
    print(f"CPU time: {elapsed_processtime}\tWall time: {elapsed_walltime}")

    faa = trainer.metric.final_average_accuracy()
    ff = trainer.metric.final_forgetting()
    print(f"FAA: {faa}")
    print(f"FF: {ff}")
    print()
    print("TRAINER.METRIC.ACCURACY")
    for k, v in trainer.metric.accuracy.items():
        print(f"{k}: {v}")
    print()
    # print(trainer.metric.accuracy)
    print("TRAINER.METRIC.FORGET")
    for k, v in trainer.metric.forget.items():
        print(f"{k}: {v}")
    # print(trainer.metric.forget)    
    print()
    if test_embedding_path:
        trainer.test_loop()

    losses = {"train_loss": train_loss, "val_loss": val_loss}
    pickle.dump(losses, open(f, "wb"))
