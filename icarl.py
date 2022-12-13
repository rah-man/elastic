import argparse
import copy
import numpy as np
import pickle
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from base import get_data, BaseDataset
from mets import Metrics
from model import BiasLayer, SingleMLP
from replay import RandomReplay, Herding
from utils import parse_input

WEIGHT_DECAY = 1e-4 # from BiC paper
lr_update_epoch = [30, 60, 80, 90]

class Trainer:
    def __init__(
        self, 
        criterion, 
        dataset,
        lr,
        wd,
        mom,
        class_order,
        batch_size=64,
        epochs=5,
        replay=None,
        metric=None,
        ):

        self.criterion = criterion
        self.dataset = dataset
        self.lr = lr
        self.wd = wd
        self.mom = mom
        self.class_order = class_order
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.replay = replay
        self.metric = metric
        self.cls2idx = {}
        self.idx2cls = {}
        self.seen_cls = 0
        self.previous_model = None
        self.class_group = []
        self.temperature = 2
        self.lambda_ = 1.6

        self.model = None
        self.train_loss = []
        self.val_loss = []

    def update_classmap(self, new_cls):
        cls_ = list(self.cls2idx.keys())
        cls_.extend(new_cls)
        self.cls2idx = {v: k for k, v in enumerate(cls_)}
        self.idx2cls = {k: v for k, v in enumerate(cls_)} 

    def train_loop(self):
        test_loaders = []
        val_loaders = []
        for task in range(len(self.dataset)):    
            new_cls = len(self.dataset[task]["classes"])
            self.class_group.append(self.dataset[task]["classes"])
            self.update_classmap(self.dataset[task]["classes"])                                      

            if self.model:
                self.previous_model = copy.deepcopy(self.model)
                self.previous_model.eval()

            self.model = self.update_model(self.seen_cls, new_cls).to(device)
            optimiser = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.mom, weight_decay=self.wd)
            scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=30)
            self.seen_cls += new_cls

            # for logging
            print(optimiser)
            print()

            x_train, y_train, x_val, y_val = self.replay._split_train_val(self.dataset[task])
            trainloader = self._get_dataloader(x_train, y_train, shuffle=True)
            val_loaders.append(self._get_dataloader(x_val, y_val, shuffle=True))
            # test_loaders.append(self._get_dataloader(self.dataset[task]["test"]["x"], self.dataset[task]["test"]["y"]))     

            replay.store_buffer(self.dataset[task])
            replay.compute_class_means()       

            print(f"Training task-{task}\nClasses: {np.unique(y_train)}")
            print(f"Class map: {self.cls2idx}")

            tloss = []
            for epoch in range(self.epochs):
                ypreds, ytrue = [], []
                self.model.train()

                running_train_loss, running_dist_loss = 0.0, 0.0
                dataset_len = 0

                for i, (image, label) in enumerate(trainloader):
                    image = image.to(self.device)
                    label = label.to(self.device)

                    outputs = self.model(image)
                    loss = self.criterion(outputs, label)

                    loss.backward()
                    optimiser.step()
                    scheduler.step()
                    optimiser.zero_grad()

                    # predicted = torch.argmax(outputs.data, 1)
                    dists = self.predict(image)
                    predicted = dists.argmin(1)
                    ypreds.extend(predicted.cpu().numpy().tolist())
                    ytrue.extend(label.cpu().numpy().tolist())  

                    running_train_loss += loss.item() * image.size(0)
                    dataset_len += image.size(0)
                
                tloss.append(running_train_loss / dataset_len)            
                self.train_loss.append(tloss)
                print(f"Epoch: {epoch+1}/{self.epochs}\ttrain_loss: {tloss[-1]:.4f}\ttrain_accuracy: {(100 * accuracy_score(ytrue, ypreds)):.4f}")
            print("FINISH TRAINING TASK-", task)
            
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
                    
                    # predicted = torch.argmax(outputs.data, 1)
                    dists = self.predict(x)                
                    predicted = dists.argmin(1)
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
            # if task < len(self.dataset) - 1:
            #     self.replay.store_buffer(self.dataset[task])
            
        
        return self.train_loss, self.val_loss

    def set_model_gradient(self, value):
        for param in self.model.parameters():
            param.requires_grad = value

    def update_model(self, seen_cls, new_cls):
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

    def _get_dataloader(self, x, y, shuffle=False):
        _dataset = BaseDataset(x, y, None, self.cls2idx)
        _dataloader = DataLoader(_dataset, batch_size=self.batch_size, shuffle=shuffle)
        return _dataloader

    # def predict(self, x):
    #     # adapted from mammoth forward(self, x)
    #     # https://github.com/aimagelab/mammoth/blob/master/models/icarl.py
    #     """
    #     x is a batch of input features, shape [batch_num, 768]

    #     return distance to the nearest class, shape [batch_num, num_classes]
    #     """
    #     feats = x.view(x.size(0), -1)
    #     feats = feats.unsqueeze(1)
    #     pred = (replay.class_means.unsqueeze(0).to(device) - feats).pow(2).sum(2)
    #     # return -pred
    #     return pred

    def predict(self, x):
        means = copy.deepcopy(replay.class_means).to(self.device)
        means = torch.stack([means] * x.size(0))
        means = means.transpose(1, 2)

        features = copy.deepcopy(x)
        features = features.unsqueeze(2)
        features = features.expand_as(means)

        dists = (features - means).pow(2).sum(1).squeeze()
        return dists

    # def compute_accuracy(model, loader, class_means):
    #     features, targets_ = utils.extract_features(model, loader)

    #     features = (features.T / (np.linalg.norm(features.T, axis=0) + EPSILON)).T

    #     # Compute score for iCaRL
    #     sqd = cdist(class_means, features, 'sqeuclidean')
    #     score_icarl = (-sqd).T

    #     return score_icarl, targets_        


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
    parser.add_argument("-mem", "--memory")
    args = parser.parse_args()

    n_task = int(args.task)
    f = args.file
    lr = float(args.learning_rate)
    wd = float(args.weight_decay)
    mom = float(args.momentum)
    mem_size = int(args.memory)
    batch = 256
    n_class = 100
    epochs = 100
    criterion = nn.CrossEntropyLoss()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data, class_order = get_data(
        train_embedding_path, 
        test_embedding_path, 
        val_embedding_path=val_embedding_path,
        num_tasks=n_task,
        # validation=0.2,
    )   

    criterion = nn.CrossEntropyLoss()
    replay = Herding(mem_size=mem_size)
    metric = Metrics()
    trainer = Trainer(criterion, data, lr, wd, mom, class_order, batch_size=batch, 
                    epochs=epochs, replay=replay, metric=metric)

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

    losses = {"train_loss": train_loss, "val_loss": val_loss}
    pickle.dump(losses, open(f, "wb"))
