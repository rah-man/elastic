import argparse
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
        epochs=5,):
    
        self.criterion = criterion
        self.dataset = dataset
        self.lr = lr       
        self.wd = wd
        self.mom = mom 
        self.total_cls = total_cls
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.train_loss = []
        self.val_loss = []
        self.train_accuracy = []
        self.val_accuracy = []

        self.model = nn.Sequential(nn.Linear(in_features=768, out_features=1000),
                                nn.ReLU(),
                                nn.Linear(in_features=1000, out_features=total_cls))

    def train_loop(self):        
        val_loaders = []        
        
        # self.dataset should have only one entry (that contains all classes)
        # i.e. task == 0
        for task in range(len(self.dataset)):  
            self.model = self.model.to(self.device)           
            optimiser = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.wd, momentum=self.mom)

            # for logging
            print(optimiser)
            print()

            trainloader, valloader = self._get_current_dataloader(task)
            
            for epoch in range(self.epochs):
                ypreds, ytrue = [], []
                self.model.train()
                running_train_loss = 0.0
                dataset_len = 0

                for i, (x, y) in enumerate(trainloader):
                    x = x.to(self.device)
                    y = y.to(self.device)

                    outputs = self.model(x)
                    loss = self.criterion(outputs, y)
                    loss.backward()
                    optimiser.step()
                    optimiser.zero_grad()
                                  
                    predicted = torch.argmax(outputs.data, 1)
                    ypreds.extend(predicted.detach().cpu().tolist())
                    ytrue.extend(y.detach().cpu().tolist())

                    running_train_loss += loss.item() * x.size(0)
                    dataset_len += x.size(0)
                
                acc = (100 * accuracy_score(ytrue, ypreds))
                self.train_loss.append(running_train_loss / dataset_len)        
                self.train_accuracy.append(acc)    
                print(f"Epoch: {epoch+1}/{self.epochs}\ttrain_loss: {self.train_loss[-1]:.4f}\ttrain_accuracy: {acc:.4f}")

                # validation step
                if valloader:
                    self.model.eval()
                    ypreds, ytrue = [], []
                    running_val_loss = 0.0
                    dataset_len = 0

                    for i, (x, y) in enumerate(trainloader):
                        x = x.to(self.device)
                        y = y.to(self.device)

                        with torch.no_grad():
                            outputs = self.model(x)
                            loss = self.criterion(outputs, y)
                                    
                        predicted = torch.argmax(outputs.data, 1)
                        ypreds.extend(predicted.detach().cpu().tolist())
                        ytrue.extend(y.detach().cpu().tolist())

                        running_val_loss += loss.item() * x.size(0)
                        dataset_len += x.size(0)

                    acc = (100 * accuracy_score(ytrue, ypreds))
                    self.val_loss.append(running_val_loss / dataset_len)            
                    self.val_accuracy.append(acc)
                    print(f"\t\tval_loss: {self.val_loss[-1]:.4f}\tval_accuracy: {acc:.4f}")
                

        return self.train_loss, self.val_loss, self.train_accuracy, self.val_accuracy

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
        _dataset = BaseDataset(x, y)
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
    parser.add_argument("-f", "--file")
    parser.add_argument("-lr", "--learning_rate")
    parser.add_argument("-wd", "--weight_decay")
    parser.add_argument("-mom", "--momentum")        
    args = parser.parse_args()
    f = args.file
    lr = float(args.learning_rate)
    wd = float(args.weight_decay)
    mom = float(args.momentum)

    batch = 256
    n_class = 100
    epochs = 100
    n_task = 1
    criterion = nn.CrossEntropyLoss()
    
    data, class_order = get_data(
        train_embedding_path, 
        test_embedding_path, 
        val_embedding_path=val_embedding_path,
        num_tasks=1, # num_tasks == n_experts
        validation=0.2,
    )

    trainer = Trainer(criterion, data, lr, wd, mom, n_class, batch_size=batch, epochs=epochs)

    walltime_start, processtime_start = time.time(), time.process_time()
    train_loss, val_loss, train_acc, val_acc = trainer.train_loop()
    walltime_end, processtime_end = time.time(), time.process_time()
    elapsed_walltime = walltime_end - walltime_start
    elapsed_processtime = processtime_end - processtime_start
    print('Execution time:', )
    print(f"CPU time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_processtime))}\tWall time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_walltime))}")
    print(f"CPU time: {elapsed_processtime}\tWall time: {elapsed_walltime}")

    print()
    if test_embedding_path:
        trainer.test_loop()

    losses = {"train_loss": train_loss, "val_loss": val_loss, "train_acc": train_acc, "val_acc": val_acc}
    pickle.dump(losses, open(f, "wb"))
