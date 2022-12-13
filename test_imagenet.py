import numpy as np
import torch
import torch.nn as nn

from collections import Counter
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from base import get_data, BaseDataset
from mets import Metrics
from replay import Herding, RandomReplay

class Simple(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=768, out_features=10)
        # self.fc2 = nn.Linear(in_features=256, out_features=128)
        # self.fc3 = nn.Linear(in_features=128, out_features=100)
        # self.relu = nn.ReLU()

    def forward(self, x):
        # return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))
        return self.fc1(x)

class Trainer:
    def __init__(self,
                criterion,
                data,
                lr,
                class_order,
                batch_size,
                epochs,
                device,
                replay,
                metric):
        self.criterion = criterion
        self.dataset = data
        self.lr = lr
        self.class_order = class_order
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.replay = replay
        self.metric = metric

        # self.model = Simple().to(self.device)   # just to check the first 10 class
        self.cls2idx = {}
        self.idx2cls = {}

    def update_classmap(self, new_cls):
        cls_ = list(self.cls2idx.keys())
        cls_.extend(new_cls)
        self.cls2idx = {v: k for k, v in enumerate(cls_)}
        self.idx2cls = {k: v for k, v in enumerate(cls_)} 

    def _get_dataloader(self, x, y, shuffle=True):
        _dataset = BaseDataset(x, y, None, self.cls2idx)
        _dataloader = DataLoader(_dataset, batch_size=self.batch_size, shuffle=shuffle)
        return _dataloader      

    def _get_current_dataloader(self, task, val=False, ignore_replay=False, uniform=False):
        current_dataset = self.dataset[task]
        if ignore_replay:
            x_train, y_train = current_dataset["train"]["x"], current_dataset["train"]["y"]
        else:
            # if uniform=False --> update the buffer from the previous tasks' datasets and append current task's dataset at the end
            # else --> update the buffer so all data from each task has uniform size
            
            # self.replay.update_buffer(current_dataset, uniform=uniform)
            self.replay.update_buffer(current_dataset, uniform=uniform)
            x_train, y_train = self.replay.buffer["x"], self.replay.buffer["y"]
            
        trainloader = self._get_dataloader(x_train, y_train, shuffle=True)

        valloader = None
        if len(current_dataset["val"]["x"]) != 0 and val:
            x_val, y_val = current_dataset["val"]["x"], current_dataset["val"]["y"]
            valloader = self._get_dataloader(x_val, y_val)

        class_counter = Counter(y_train)
        print("CLASS_COUNTER:", class_counter)

        return trainloader, valloader

    def train_loop(self):
        self.model = Simple().to(device)
        for task in range(len(self.dataset)):
            self.update_classmap(self.dataset[task]["classes"])        
            # optimiser = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)         
            optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
            x_train, y_train, x_val, y_val = self.replay._split_train_val(self.dataset[task])
            trainloader = self._get_dataloader(x_train, y_train, shuffle=True)
            valloader = self._get_dataloader(x_val, y_val, shuffle=True)       
            # trainloader, valloader = self._get_current_dataloader(task, val=True, ignore_replay=True) 

            for epoch in range(self.epochs):
                train_loss, val_loss = [], []
                ypreds, ytrue = [], []

                for images, labels in trainloader:
                    images = images.to(device)
                    labels = labels.to(device)                    

                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    train_loss.append(loss.item())
                    loss.backward()
                    optimiser.step()
                    optimiser.zero_grad()

                    predicted = torch.argmax(outputs.data, 1)
                    # print(f"\tPREDICTED: {predicted.cpu().numpy().tolist()}")
                    # print(f"\tTRUE: {labels.cpu().numpy().tolist()}")
                    ypreds.extend(predicted.cpu().numpy().tolist())
                    ytrue.extend(labels.cpu().numpy().tolist())
                print(f"Epoch: {epoch+1}/{self.epochs}\tloss: {np.average(train_loss):.4f}\ttrain_accuracy: {(100 * accuracy_score(ytrue, ypreds)):.4f}")
            break

if __name__ == "__main__":
    train_embedding_path = "cifar100_train_embedding.pt"
    test_embedding_path = "cifar100_test_embedding.pt"
    val_embedding_path = None

    batch = 64
    n_class = 100
    steps = 1
    mem_size = 2000
    epochs = 20
    n_task = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data, class_order = get_data(
        train_embedding_path, 
        test_embedding_path, 
        val_embedding_path=val_embedding_path,
        num_tasks=n_task,
        validation=0.2,
    )   

    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    # random_replay = RandomReplay(mem_size=mem_size, mode="icarl")
    replay = Herding(mem_size=mem_size)
    metric = Metrics()
    trainer = Trainer(criterion, data, lr, class_order, batch_size=batch, 
                    epochs=epochs, device=device, replay=replay, metric=metric)
    trainer.train_loop()

# import copy
# import getopt
# import numpy as np
# import sys
# import time
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision.models as models

# from collections import Counter
# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision.models.feature_extraction import create_feature_extractor

# from base import BaseDataset, Extractor, get_data
# from dynamicmoe import DynamycMoE
# from mets import Metrics
# from model import SingleMLP
# from regulariser import LwF, Regulariser
# from replay import RandomReplay
# from sklearn.metrics import accuracy_score


# WEIGHT_DECAY = 5e-4 # from LwF paper
# # WEIGHT_DECAY = 1e-4 # from BiC paper
# lr_update_epoch = [30, 60, 80, 90]

# class Trainer:
#     def __init__(
#         self, 
#         criterion, 
#         dataset,
#         lr,
#         total_cls,
#         class_order,
#         extractor=None,
#         batch_size=64,
#         epochs=5,
#         device="cpu",        
#         replay=None,
#         transform=None,
#         print_every=50,
#         mode=None,
#         k=None,
#         metric=None,
#         regulariser=None):
    
#         self.criterion = criterion
#         self.dataset = dataset
#         self.extractor = extractor
#         self.lr = lr        
#         self.total_cls = total_cls
#         self.class_order = class_order
#         self.batch_size = batch_size
#         self.epochs = epochs
#         self.device = device
#         self.replay = replay
#         self.transform = transform
#         self.print_every = print_every
#         self.mode = mode
#         self.k = k
#         self.metric = metric
    

#         self.seen_cls = 0
#         # self.model = None
#         self.previous_model = None
#         self.previous_cls2idx = None
#         self.previous_idx2cls = None
#         # self.cls2idx = {v: k for k, v in enumerate(self.class_order)}
#         # self.idx2cls = {k: v for k, v in enumerate(self.class_order)}
#         self.cls2idx = {}
#         self.idx2cls = {}

#         if mode == "expert":
#             self.model = DynamycMoE(k=self.k)

#     def update_classmap(self, new_cls):
#         cls_ = list(self.cls2idx.keys())
#         cls_.extend(new_cls)
#         self.cls2idx = {v: k for k, v in enumerate(cls_)}
#         self.idx2cls = {k: v for k, v in enumerate(cls_)}

#     def train_loop(self, steps=2):        
#         val_loaders = []        
        
#         for task in range(len(self.dataset)):  
#             # update the model for new classes in each iteration
#             new_cls = len(self.dataset[task]["classes"])
#             self.update_classmap(self.dataset[task]["classes"])            

#             if self.mode == "expert":
#                 self.model.expand_expert(self.seen_cls, new_cls, self.k)                
#                 # print(self.model)

#             print(f"TRAINING TASK-{task}\tCLASSES: {self.dataset[task]['classes']}")

#             self.model = self.model.to(self.device)           
#             # optimiser = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=WEIGHT_DECAY)
#             optimiser = optim.SGD(self.model.parameters(), lr=0.001, weight_decay=WEIGHT_DECAY)
            
#             if steps == 2:
#                 # freeze previous experts and train using Di only --> ignore_replay=True
#                 # NOTE: for 2 steps: ignore_replay must be set to True                            
#                 self.model.freeze_previous()            
#                 trainloader, valloader = self._get_current_dataloader(task, val=True, ignore_replay=True) 
#                 # val_loaders.append(valloader)
#             elif steps == 1:
#                 # NOTE: for 1 step: ignore_replay must be set to False (i.e. the default) and hide the second step
#                 trainloader, valloader = self._get_current_dataloader(task, val=True) 
#                 # val_loaders.append(valloader)
#             val_loaders.append(valloader)

#             for epoch in range(self.epochs):
#                 train_loss, val_loss = [], []
#                 ypreds, ytrue = [], []
#                 self.model.train()

#                 # print(f"Epoch: {epoch+1}/{self.epochs}")

#                 for i, (x, y) in enumerate(trainloader):
#                     x = x.to(self.device)
#                     y = y.to(self.device)

#                     outputs, gate_loss, expert_losses, _ = self.model(x, y)
#                     loss = self.criterion(outputs, y)

#                     total_loss = loss + gate_loss + sum(expert_losses)
#                     total_loss.backward()
#                     optimiser.step()
#                     optimiser.zero_grad()

#                     train_loss.append(loss.item())
#                     predicted = torch.argmax(outputs.data, 1)
#                     ypreds.extend(predicted.detach().cpu().tolist())
#                     ytrue.extend(y.detach().cpu().tolist())

#                 print(f"Epoch: {epoch+1}/{self.epochs}\tloss: {np.average(train_loss):.4f}\ttrain_accuracy: {(100 * accuracy_score(ytrue, ypreds)):.4f}")

#                 # if (epoch+1) in lr_update_epoch:
#                 #     cur_lr = None
#                 #     for param_group in optimiser.param_groups:
#                 #         cur_lr = param_group["lr"]
#                 #     print(f"\tepoch: {epoch+1}")
#                 #     print(f"\tcur_lr: {cur_lr}")
#                 #     for param_group in optimiser.param_groups:
#                 #         param_group["lr"] = param_group["lr"] / 10
#                 #     new_lr = None
#                 #     for param_group in optimiser.param_groups:
#                 #         new_lr = param_group["lr"]
#                 #     print(f"\tnew_lr: {new_lr}")                

#             print("FINISH STEP 1")

#             if steps == 2:
#                 # freeze all and train using a uniform size dataset
#                 # optimiser = optim.SGD(self.model.parameters(), lr=0.001, weight_decay=WEIGHT_DECAY)
#                 self.model.freeze_all()
#                 # optimiser = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=WEIGHT_DECAY)                
#                 trainloader, _ = self._get_current_dataloader(task, uniform=True)
#                 # for epoch in range(self.epochs):
#                 step2_epochs = 20
#                 for epoch in range(step2_epochs):
#                     train_loss, val_loss = [], []
#                     ypreds, ytrue = [], []
#                     self.model.train()

#                     # print(f"Epoch: {epoch+1}/{self.epochs}")

#                     for i, (x, y) in enumerate(trainloader):
#                         x = x.to(self.device)
#                         y = y.to(self.device)

#                         outputs, gate_loss, expert_losses, _ = self.model(x, y)
#                         loss = self.criterion(outputs, y)
#                         total_loss = loss + gate_loss + sum(expert_losses)
        
#                         total_loss.backward()
#                         optimiser.step()
#                         optimiser.zero_grad()

#                         train_loss.append(loss.item())
#                         predicted = torch.argmax(outputs.data, 1)
#                         ypreds.extend(predicted.detach().cpu().tolist())
#                         ytrue.extend(y.detach().cpu().tolist())

#                     print(f"Epoch: {epoch+1}/{step2_epochs}\tloss: {np.average(train_loss):.4f}\ttrain_accuracy: {(100 * accuracy_score(ytrue, ypreds)):.4f}")            

#                     # if (epoch+1) in lr_update_epoch:
#                     #     cur_lr = None
#                     #     for param_group in optimiser.param_groups:
#                     #         cur_lr = param_group["lr"]
#                     #     print(f"\tepoch: {epoch+1}")
#                     #     print(f"\tcur_lr: {cur_lr}")
#                     #     for param_group in optimiser.param_groups:
#                     #         param_group["lr"] = param_group["lr"] / 10
#                     #     new_lr = None
#                     #     for param_group in optimiser.param_groups:
#                     #         new_lr = param_group["lr"]
#                     #     print(f"\tnew_lr: {new_lr}")                

#                 print("FINISH STEP 2")

#             self.model.unfreeze_all()

#             if val_loaders:
#                 self.model.eval()
#                 for i, valloader in enumerate(val_loaders):
#                     ypreds, ytrue = [], []
#                     for x, y in valloader:
#                         x = x.to(self.device)
#                         y = y.to(self.device)

#                         with torch.no_grad():
#                             outputs, gate_loss, expert_losses, _ = self.model(x, y)
#                             loss = self.criterion(outputs, y)
#                             total_loss = loss + gate_loss + sum(expert_losses)
#                             val_loss.append(loss.item())
#                         predicted = torch.argmax(outputs.data, 1)
#                         ypreds.extend(predicted.detach().cpu().tolist())
#                         ytrue.extend(y.detach().cpu().tolist())
#                 # print(f"\tVal Loss: {np.average(val_loss):.4f}\tval_accuracy: {(100 * accuracy_score(ytrue, ypreds)):.4f}")
#                     task_accuracy = 100 * accuracy_score(ytrue, ypreds)
#                     print(f"\tTask-{i} val_accuracy: {task_accuracy:.4f}")
#                     if self.metric:
#                         self.metric.add_accuracy(task, task_accuracy)
#             # self.test_loop(task)
#             if self.metric:
#                 self.metric.add_forgetting(task)
#             print()

#             self.seen_cls += new_cls

#     def test_loop(self, current_task=None):
#         self.model.to(self.device)
#         self.model.eval()
#         print("CALCULATING TEST LOSS PER TASK")

#         num_test = len(self.dataset) if current_task is None else current_task + 1     
#         all_preds, all_true = [], []   
#         for task in range(num_test):
#             ypreds, ytrue = [], []
#             x_test, y_test = self.dataset[task]["test"]["x"], self.dataset[task]["test"]["y"]
#             testloader = self._get_dataloader(x_test, y_test)
            
#             for x, y in testloader:
#                 x = x.to(self.device)
#                 y = y.to(self.device)

#                 with torch.no_grad():
#                     outputs, _, _, _ = self.model(x, y)                    
#                 predicted = torch.argmax(outputs.data, 1)
#                 ypreds.extend(predicted.detach().cpu().tolist())
#                 ytrue.extend(y.detach().cpu().tolist())
                
#                 all_preds.extend(predicted.detach().cpu().tolist())
#                 all_true.extend(y.detach().cpu().tolist())

#             print(f"\tTASK-{task}\tCLASSES: {self.dataset[task]['classes']}\ttest_accuracy: {(100 * accuracy_score(ytrue, ypreds)):.4f}")
#         print(f"All accuracy: {(100 * accuracy_score(all_true, all_preds)):.4f}")

#     def _get_current_dataloader(self, task, val=False, ignore_replay=False, uniform=False):
#         current_dataset = self.dataset[task]
#         if ignore_replay:
#             x_train, y_train = current_dataset["train"]["x"], current_dataset["train"]["y"]
#         else:
#             # if uniform=False --> update the buffer from the previous tasks' datasets and append current task's dataset at the end
#             # else --> update the buffer so all data from each task has uniform size
            
#             # self.replay.update_buffer(current_dataset, uniform=uniform)
#             self.replay.update_buffer(current_dataset, uniform=uniform)
#             x_train, y_train = self.replay.buffer["x"], self.replay.buffer["y"]
            
#         trainloader = self._get_dataloader(x_train, y_train, shuffle=True)

#         valloader = None
#         if len(current_dataset["val"]["x"]) != 0 and val:
#             x_val, y_val = current_dataset["val"]["x"], current_dataset["val"]["y"]
#             valloader = self._get_dataloader(x_val, y_val)

#         class_counter = Counter(y_train)
#         print("CLASS_COUNTER:", class_counter)

#         return trainloader, valloader
    
#     def _get_dataloader(self, x, y, shuffle=False):
#         _dataset = BaseDataset(x, y, self.transform, self.cls2idx)
#         _dataloader = DataLoader(_dataset, batch_size=self.batch_size, shuffle=shuffle)
#         return _dataloader

# def parse_input(argv):
#     batch = 256
#     n_class = 0
#     steps = 0
#     mem_size = 0
#     epochs = 0
#     n_experts = 0
#     k = 0

#     arg_help = "{0} -b <batch> -c <n_class> -s <steps> -m <mem_size> -e <epochs> -n <n_experts> -k <k>".format(argv[0])

#     try:
#         opts, args = getopt.getopt(argv[1:], "hb:c:s:m:e:n:k:", ["help", "batch=", "n_class=", "steps=", "mem_size=", "epochs=", "n_experts=", "k="])
#     except:
#         print(arg_help)
#         sys.exit(2)
        
#     for opt, arg in opts:
#         if opt in ("-h", "--help"):
#             print(arg_help)  # print the help message
#             sys.exit(2)
#         elif opt in ("-b", "--batch"):
#             batch = int(arg)
#         elif opt in ("-c", "--n_class"):
#             n_class = int(arg)
#         elif opt in ("-s", "--steps"):
#             steps = int(arg)
#         elif opt in ("-m", "--mem_size"):
#             mem_size = int(arg)
#         elif opt in ("-e", "--epochs"):
#             epochs = int(arg)
#         elif opt in ("-n", "--n_experts"):
#             n_experts = int(arg)
#         elif opt in ("-k", "--k"):
#             k = int(arg)

#     return batch, n_class, steps, mem_size, epochs, n_experts, k

# if __name__ == "__main__":
#     lr = 0.001
#     n_class = 100
#     criterion = nn.CrossEntropyLoss()

#     # the following dataset and embedding pointers are not needed anymore
#     # dataset = datasets.CIFAR10
#     # data_path = "CIFAR_data/"
#     # base_extractor = models.vit_b_16
#     # weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
#     # transform = weights.transforms()
#     # return_nodes = ["getitem_5"]

#     # train_embedding_path = "cifar10_train_embedding.pt"
#     # test_embedding_path = "cifar10_test_embedding.pt"

#     # train_embedding_path = "cifar100_coarse_train_embedding.pt"
#     # test_embedding_path = "cifar100_coarse_test_embedding.pt"    

#     train_embedding_path = "cifar100_train_embedding.pt"
#     test_embedding_path = "cifar100_test_embedding.pt"
#     val_embedding_path = None

#     # train_embedding_path = "imagenet100_train_embedding.pt"
#     # test_embedding_path = None

#     # train_embedding_path = "imagenet1000_train_embedding.pt"
#     # test_embedding_path = None
#     # val_embedding_path = "imagenet1000_val_embedding.pt"
    
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # batch, n_class, steps, mem_size, epochs, n_experts, k = parse_input(sys.argv)
#     num_classes = 100
#     steps = 2    
#     mem_size = 2000
#     epochs = 20
#     n_experts = 20
#     k = 5
#     batch = 64
    

#     random_replay = RandomReplay(mem_size=mem_size)
#     data, class_order = get_data(
#         train_embedding_path, 
#         test_embedding_path, 
#         val_embedding_path=val_embedding_path,
#         num_tasks=n_experts, # num_tasks == n_experts
#         validation=0.2,
#         )
    
#     met = Metrics()
#     trainer = Trainer(
#         criterion, data, lr, 
#         n_class, class_order, 
#         batch_size=batch, 
#         epochs=epochs, 
#         device=device, 
#         replay=random_replay, 
#         mode="expert", 
#         k=k, 
#         metric=met,)

#     walltime_start, processtime_start = time.time(), time.process_time()
#     trainer.train_loop(steps=steps)
#     walltime_end, processtime_end = time.time(), time.process_time()
#     elapsed_walltime = walltime_end - walltime_start
#     elapsed_processtime = processtime_end - processtime_start
#     print('Execution time:', )
#     print(f"CPU time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_processtime))}\tWall time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_walltime))}")
#     print(f"CPU time: {elapsed_processtime}\tWall time: {elapsed_walltime}")

#     exit()

#     faa = trainer.metric.final_average_accuracy()
#     ff = trainer.metric.final_forgetting()
#     print(f"FAA: {faa}")
#     print(f"FF: {ff}")
#     print()
#     print("TRAINER.METRIC.ACCURACY")
#     for k, v in trainer.metric.accuracy.items():
#         print(f"{k}: {v}")
#     print()
#     # print(trainer.metric.accuracy)
#     print("TRAINER.METRIC.FORGET")
#     for k, v in trainer.metric.forget.items():
#         print(f"{k}: {v}")
#     # print(trainer.metric.forget)    
#     print()
#     if test_embedding_path:
#         trainer.test_loop()
