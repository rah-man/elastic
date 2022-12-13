import copy
import getopt
import numpy as np
import pickle
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from collections import Counter
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models.feature_extraction import create_feature_extractor

from base import BaseDataset, Extractor, get_data
from dynamicmoe import DynamycMoE
from mets import Metrics
from model import SingleMLP
from regulariser import LwF, Regulariser
from replay import RandomReplay
from sklearn.metrics import accuracy_score


WEIGHT_DECAY = 5e-4 # from LwF paper
# WEIGHT_DECAY = 1e-4 # from BiC paper
lr_update_epoch = [30, 60, 80, 90]

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
        transform=None,
        print_every=50,
        mode=None,
        k=None,
        metric=None,
        regulariser=None):
    
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
        self.mode = mode
        self.k = k
        self.metric = metric
    

        self.seen_cls = 0
        # self.model = None
        self.previous_model = None
        self.previous_cls2idx = None
        self.previous_idx2cls = None
        # self.cls2idx = {v: k for k, v in enumerate(self.class_order)}
        # self.idx2cls = {k: v for k, v in enumerate(self.class_order)}
        self.cls2idx = {}
        self.idx2cls = {}
        self.train_loss1 = [] # for losses when training step 1
        self.train_loss2 = [] # for losses when training step 2
        self.val_loss = []

        if mode == "expert":
            self.model = DynamycMoE(k=self.k)

    def update_classmap(self, new_cls, task=None):
        cls_ = list(self.cls2idx.keys())
        if task == 1:
            # only do this once for experiment
            cls_.extend(self.dataset[task-1]["classes"])

        cls_.extend(new_cls)
        self.cls2idx = {v: k for k, v in enumerate(cls_)}
        self.idx2cls = {k: v for k, v in enumerate(cls_)}

    def train_loop(self, steps=2):        
        val_loaders = []        
        
        # for task in range(len(self.dataset)):  
        # 20221121 - EXPERIMENTAL with 2 tasks at creation
        for task in range(1, len(self.dataset)):
            # update the model for new classes in each iteration
            new_cls = len(self.dataset[task]["classes"])
            self.update_classmap(self.dataset[task]["classes"], task=task)

            if self.mode == "expert":
                self.model.expand_expert(self.seen_cls, new_cls, self.k)
                # print(self.model)

            print(self.model)

            print(f"TRAINING TASK-{task}\tCLASSES: {self.dataset[task]['classes']}")

            self.model = self.model.to(self.device)           
            # optimiser = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=WEIGHT_DECAY)
            optimiser = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=WEIGHT_DECAY)
            
            if steps == 2:
                # freeze previous experts and train using Di only --> ignore_replay=True
                # NOTE: for 2 steps: ignore_replay must be set to True                            
                self.model.freeze_previous()            
                trainloader, valloader = self._get_current_dataloader(task, val=True, ignore_replay=True) 
                # val_loaders.append(valloader)
            elif steps == 1:
                # NOTE: for 1 step: ignore_replay must be set to False (i.e. the default) and hide the second step
                trainloader, valloader = self._get_current_dataloader(task, val=True) 
                # val_loaders.append(valloader)
            val_loaders.append(valloader)

            train_loss = [] # stores the train_loss per epoch
            for epoch in range(self.epochs):
                ypreds, ytrue = [], []
                self.model.train()

                # print(f"Epoch: {epoch+1}/{self.epochs}")
                running_train_loss = 0.0
                dataset_len = 0

                for i, (x, y) in enumerate(trainloader):
                    x = x.to(self.device)
                    y = y.to(self.device)

                    outputs, gate_loss, expert_losses, _ = self.model(x, y)
                    loss = self.criterion(outputs, y)

                    total_loss = loss + gate_loss + sum(expert_losses)
                    total_loss.backward()
                    optimiser.step()
                    optimiser.zero_grad()
                                  
                    predicted = torch.argmax(outputs.data, 1)
                    ypreds.extend(predicted.detach().cpu().tolist())
                    ytrue.extend(y.detach().cpu().tolist())

                    running_train_loss += loss.item() * x.size(0)
                    dataset_len += x.size(0)
                
                train_loss.append(running_train_loss / dataset_len)            
                print(f"STEP-1\tEpoch: {epoch+1}/{self.epochs}\tloss: {train_loss[-1]:.4f}\ttrain_accuracy: {(100 * accuracy_score(ytrue, ypreds)):.4f}")

            self.train_loss1.append(train_loss)
                # if (epoch+1) in lr_update_epoch:
                #     self.reduce_learning_rate(optimiser)

            print("FINISH STEP 1")

            if steps == 2:
                # freeze all and train using a uniform size dataset
                self.model.freeze_all()

                # CHECK IF THIS CODE MAKES ANY DIFFERENCE
                # optimiser = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=WEIGHT_DECAY)
                
                trainloader, _ = self._get_current_dataloader(task, uniform=True)
                train_loss = [] # store train_loss per epoch
                for epoch in range(self.epochs):
                # step2_epochs = 
                # for epoch in range(step2_epochs):
                    running_train_loss = 0.0
                    dataset_len = 0
                    ypreds, ytrue = [], []
                    self.model.train()

                    # print(f"Epoch: {epoch+1}/{self.epochs}")

                    for i, (x, y) in enumerate(trainloader):
                        x = x.to(self.device)
                        y = y.to(self.device)

                        outputs, gate_loss, expert_losses, _ = self.model(x, y)
                        loss = self.criterion(outputs, y)
                        total_loss = loss + gate_loss + sum(expert_losses)
        
                        total_loss.backward()
                        optimiser.step()
                        optimiser.zero_grad()

                        predicted = torch.argmax(outputs.data, 1)
                        ypreds.extend(predicted.detach().cpu().tolist())
                        ytrue.extend(y.detach().cpu().tolist())

                        running_train_loss += loss.item() * x.size(0)
                        dataset_len += x.size(0)
                    
                    train_loss.append(running_train_loss / dataset_len)
                    print(f"STEP-2\tEpoch: {epoch+1}/{self.epochs}\tloss: {train_loss[-1]:.4f}\ttrain_accuracy: {(100 * accuracy_score(ytrue, ypreds)):.4f}")      

                self.train_loss2.append(train_loss)
                    # if (epoch+1) in lr_update_epoch:
                    #     self.reduce_learning_rate(optimiser)                  

                print("FINISH STEP 2")

            self.model.unfreeze_all()

            if val_loaders:
                self.model.eval()
                                
                val_loss_ = []
                for i, valloader in enumerate(val_loaders):
                    running_val_loss = 0.0
                    dataset_len = 0
                    ypreds, ytrue = [], []
                    for x, y in valloader:
                        x = x.to(self.device)
                        y = y.to(self.device)

                        with torch.no_grad():
                            outputs, gate_loss, expert_losses, _ = self.model(x, y)
                            loss = self.criterion(outputs, y)
                            total_loss = loss + gate_loss + sum(expert_losses)                            
                            running_val_loss += loss.item() * x.size(0)
                        predicted = torch.argmax(outputs.data, 1)
                        ypreds.extend(predicted.detach().cpu().tolist())
                        ytrue.extend(y.detach().cpu().tolist())
                        dataset_len += x.size(0)

                    
                    val_loss_.append(running_val_loss / dataset_len)
                # print(f"\tVal Loss: {np.average(val_loss):.4f}\tval_accuracy: {(100 * accuracy_score(ytrue, ypreds)):.4f}")
                    
                    task_accuracy = 100 * accuracy_score(ytrue, ypreds)
                    print(f"\tTask-{i} val_loss: {val_loss_[-1]:.4f}\tval_accuracy: {task_accuracy:.4f}")
                    if self.metric:
                        self.metric.add_accuracy(task, task_accuracy)
                self.val_loss.append(val_loss_)
            # self.test_loop(task)
            if self.metric:
                self.metric.add_forgetting(task)
            print()

            self.seen_cls += new_cls

        return self.train_loss1, self.train_loss2, self.val_loss

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
                    outputs, _, _, _ = self.model(x, y)                    
                predicted = torch.argmax(outputs.data, 1)
                ypreds.extend(predicted.detach().cpu().tolist())
                ytrue.extend(y.detach().cpu().tolist())
                
                all_preds.extend(predicted.detach().cpu().tolist())
                all_true.extend(y.detach().cpu().tolist())

            print(f"\tTASK-{task}\tCLASSES: {self.dataset[task]['classes']}\ttest_accuracy: {(100 * accuracy_score(ytrue, ypreds)):.4f}")
        print(f"All accuracy: {(100 * accuracy_score(all_true, all_preds)):.4f}")

    def _get_current_dataloader(self, task, val=False, ignore_replay=False, uniform=False):
        current_dataset = self.dataset[task]
        if ignore_replay and task == 1:
            current_dataset = self.dataset[task-1]
            x_train, y_train = current_dataset["train"]["x"], current_dataset["train"]["y"]
            current_dataset = self.dataset[task]
            x_train.extend(current_dataset["train"]["x"])
            y_train.extend(current_dataset["train"]["y"])
        elif ignore_replay and task != 1:
            x_train, y_train = current_dataset["train"]["x"], current_dataset["train"]["y"]
        elif uniform == True and task == 1:
            prev_dataset = self.dataset[task-1]
            self.replay.update_buffer(current_dataset, uniform=uniform, task=task, prev_dataset=prev_dataset)
            x_train, y_train = self.replay.buffer["x"], self.replay.buffer["y"]            
        else:
            # if uniform=False --> update the buffer from the previous tasks' datasets and append current task's dataset at the end
            # else --> update the buffer so all data from each task has uniform size
            
            # self.replay.update_buffer(current_dataset, uniform=uniform)
            self.replay.update_buffer(current_dataset, uniform=uniform)
            x_train, y_train = self.replay.buffer["x"], self.replay.buffer["y"]
            
        trainloader = self._get_dataloader(x_train, y_train, shuffle=True)

        valloader = None
        if task == 1 and len(current_dataset["val"]["x"]) != 0 and val:
            current_dataset = self.dataset[task-1]
            x_val, y_val = current_dataset["val"]["x"], current_dataset["val"]["y"]
            current_dataset = self.dataset[task]
            x_val.extend(current_dataset["val"]["x"])
            y_val.extend(current_dataset["val"]["y"])
            valloader = self._get_dataloader(x_val, y_val)
        elif task != 1 and len(current_dataset["val"]["x"]) != 0 and val:
            x_val, y_val = current_dataset["val"]["x"], current_dataset["val"]["y"]
            valloader = self._get_dataloader(x_val, y_val)

        class_counter = Counter(y_train)
        print("CLASS_COUNTER:", class_counter)

        return trainloader, valloader
    
    def _get_dataloader(self, x, y, shuffle=False):
        _dataset = BaseDataset(x, y, self.transform, self.cls2idx)
        _dataloader = DataLoader(_dataset, batch_size=self.batch_size, shuffle=shuffle)
        return _dataloader

    def reduce_learning_rate(self, optimiser):        
        for param_group in optimiser.param_groups:
            old_lr = param_group["lr"]
        for param_group in optimiser.param_groups:
            param_group["lr"] = param_group["lr"] / 10
            new_lr = param_group["lr"]
        print(f"Reduce learning_rate from {old_lr} to {new_lr}")

def parse_input(argv):
    batch = 256
    n_class = 0
    steps = 0
    mem_size = 0
    epochs = 0
    n_experts = 0
    k = 0

    arg_help = "{0} -b <batch> -c <n_class> -s <steps> -m <mem_size> -e <epochs> -n <n_experts> -k <k> -f <f>".format(argv[0])

    try:
        opts, args = getopt.getopt(argv[1:], "hb:c:s:m:e:n:k:f:", ["help", "batch=", "n_class=", "steps=", "mem_size=", "epochs=", "n_experts=", "k=", "f="])
    except:
        print(arg_help)
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # print the help message
            sys.exit(2)
        elif opt in ("-b", "--batch"):
            batch = int(arg)
        elif opt in ("-c", "--n_class"):
            n_class = int(arg)
        elif opt in ("-s", "--steps"):
            steps = int(arg)
        elif opt in ("-m", "--mem_size"):
            mem_size = int(arg)
        elif opt in ("-e", "--epochs"):
            epochs = int(arg)
        elif opt in ("-n", "--n_experts"):
            n_experts = int(arg)
        elif opt in ("-k", "--k"):
            k = int(arg)
        elif opt in ("-f", "--file"):
            f = arg

    return batch, n_class, steps, mem_size, epochs, n_experts, k, f

if __name__ == "__main__":
    lr = 0.001
    n_class = 20
    criterion = nn.CrossEntropyLoss()

    # the following dataset and embedding pointers are not needed anymore
    # dataset = datasets.CIFAR10
    # data_path = "CIFAR_data/"
    # base_extractor = models.vit_b_16
    # weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
    # transform = weights.transforms()
    # return_nodes = ["getitem_5"]

    # train_embedding_path = "cifar10_train_embedding.pt"
    # test_embedding_path = "cifar10_test_embedding.pt"

    # train_embedding_path = "cifar100_coarse_train_embedding.pt"
    # test_embedding_path = None
    # val_embedding_path = "cifar100_coarse_test_embedding.pt"    

    train_embedding_path = "cifar100_train_embedding.pt"
    test_embedding_path = None
    val_embedding_path = "cifar100_test_embedding.pt"

    # train_embedding_path = "imagenet100_train_embedding.pt"
    # test_embedding_path = None

    # train_embedding_path = "imagenet1000_train_embedding.pt"
    # test_embedding_path = None
    # val_embedding_path = "imagenet1000_val_embedding.pt"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch, n_class, steps, mem_size, epochs, n_experts, k, f = parse_input(sys.argv)
    # num_classes = 100
    # steps = 2    
    # mem_size = 2000
    # epochs = 20
    # n_experts = 20
    # k = 5
    

    random_replay = RandomReplay(mem_size=mem_size)
    data, class_order = get_data(
        train_embedding_path, 
        test_embedding_path, 
        val_embedding_path=val_embedding_path,
        num_tasks=n_experts, # num_tasks == n_experts
        # validation=0.1,
    )
    
    met = Metrics()
    trainer = Trainer(
        criterion, data, lr, 
        n_class, class_order, 
        batch_size=batch, 
        epochs=epochs, 
        device=device, 
        replay=random_replay, 
        mode="expert", 
        k=k, 
        metric=met,)

    walltime_start, processtime_start = time.time(), time.process_time()
    train_loss1, train_loss2, val_loss = trainer.train_loop(steps=steps)
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

    losses = {"train_loss1": train_loss1, "train_loss2": train_loss2, "val_loss": val_loss, "metric_acc": met.accuracy, "metric_forget": met.forget}
    pickle.dump(losses, open(f, "wb"))
