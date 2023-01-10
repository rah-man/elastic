import argparse
import copy
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import wandb

from collections import Counter
from torch.utils.data import DataLoader
from torchvision.models.feature_extraction import create_feature_extractor

from base import BaseDataset, get_data
from dynamicmoe import DynamycMoE, BiasLayer
from mets import Metrics
from replay import RandomReplay
from sklearn.metrics import accuracy_score


WEIGHT_DECAY = 5e-4

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
        k=None,
        metric=None,):
    
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
        self.transform = None
        self.k = k
        self.metric = metric
    
        self.seen_cls = 0
        self.previous_model = None
        self.previous_cls2idx = None
        self.previous_idx2cls = None
        self.cls2idx = {}
        self.idx2cls = {}
        self.train_loss1 = [] # for losses when training step 1
        self.train_loss2 = [] # for losses when training step 2
        self.val_loss = []
        self.lambda_= 1.6
        self.temperature = 2
        # self.bias_layers = []
        self.previous_task_nums = []

        self.model = DynamycMoE(k=self.k)

    def update_classmap(self, new_cls):
        cls_ = list(self.cls2idx.keys())
        cls_.extend(new_cls)
        self.cls2idx = {v: k for k, v in enumerate(cls_)}
        self.idx2cls = {k: v for k, v in enumerate(cls_)}

    def train_loop(self, steps=2):        
        val_loaders = []        
                
        for task in range(len(self.dataset) - 1): # -1 because there is an additional key: ncla  
            # update the model for new classes in each iteration
            new_cls = len(self.dataset[task]["classes"])
            self.update_classmap(self.dataset[task]["classes"])
            # self.bias_layers.append(BiasLayer().to(self.device))

            self.model.expand_expert(self.seen_cls, new_cls, self.k)
            if task > 0:
                self.previous_model = copy.deepcopy(self.model)
                self.previous_model.to(device)
                self.previous_model.eval()            

            self.model.to(self.device)            
            optimiser = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=WEIGHT_DECAY, momentum=0.5)
            
            if steps == 2:
                # freeze previous experts and train using Di only --> ignore_replay=True
                # NOTE: for 2 steps: ignore_replay must be set to True                            
                self.model.freeze_previous()    
                # self.model.set_gate(False)        
                trainloader, valloader = self._get_current_dataloader(task, val=True, ignore_replay=True) 
            elif steps == 1:
                # NOTE: for 1 step: ignore_replay must be set to False (i.e. the default) and hide the second step
                trainloader, valloader = self._get_current_dataloader(task, val=True) 
            val_loaders.append(valloader)

            print("STARTING STEP 1")
            for name, param in self.model.named_parameters():
                print(name, param.requires_grad)            
            train_loss = [] # stores the train_loss per epoch
            # print(self.model)
            for epoch in range(self.epochs):
                ypreds, ytrue = [], []
                self.model.train()

                # print(f"Epoch: {epoch+1}/{self.epochs}")
                running_train_loss = 0.0
                dataset_len = 0

                for x, y in trainloader:
                    x = x.to(self.device)
                    y = y.to(self.device)

                    outputs, gate_loss, expert_losses, gates, expert_outputs = self.model(x, y)

                    if task > 0:
                        bias_outputs = []
                        prev = 0
                        for i, num_class in enumerate(self.previous_task_nums):
                            outs = outputs[:, prev:(prev+num_class)]
                            bias_outputs.append(self.model.bias_forward(task, outs))
                            prev += num_class
                        old_cls_outputs = torch.cat(bias_outputs, dim=1)
                        new_cls_outputs = self.model.bias_layers[task](outputs[:, prev:])                    
                        pred_all_classes = torch.cat([old_cls_outputs, new_cls_outputs], dim=1)
                        loss = self.criterion(pred_all_classes, y)
                        loss += 0.1 * ((self.model.bias_layers[task].beta[0] ** 2) / 2)
                    else:
                        loss = self.criterion(outputs, y)

                    total_loss = loss + gate_loss #+ sum(expert_losses)
                    total_loss.backward()
                    optimiser.step()
                    optimiser.zero_grad()
                                  
                    predicted = torch.argmax(outputs.data, 1) if task == 0 else torch.argmax(pred_all_classes.data, 1)
                    ypreds.extend(predicted.detach().cpu().tolist())
                    ytrue.extend(y.detach().cpu().tolist())

                    running_train_loss += loss.item() * x.size(0)
                    dataset_len += x.size(0)
                
                # scheduler.step()
                train_loss.append(running_train_loss / dataset_len)            
                print(f"STEP-1\tEpoch: {epoch+1}/{self.epochs}\tloss: {train_loss[-1]:.4f}\tstep1_train_accuracy: {(100 * accuracy_score(ytrue, ypreds)):.4f}")

            self.train_loss1.append(train_loss)

            print("FINISH STEP 1")

            if steps == 2:
                print("STARTING STEP 2")
                # freeze all and train using a uniform size dataset
                self.model.freeze_all()
                # self.model.set_gate(True)

                for name, param in self.model.named_parameters():
                    print(name, param.requires_grad)

                trainloader, _ = self._get_current_dataloader(task, uniform=True)

                train_loss = [] # store train_loss per epoch

                self.model.bias_layers[task].alpha.requires_grad = True
                self.model.bias_layers[task].beta.requires_grad = True
                bias_optimiser = optim.SGD(self.model.parameters(), lr=0.001)
                step2_epochs = 200
                # print(self.model)
                for epoch in range(step2_epochs):
                    running_train_loss = 0.0
                    dataset_len = 0
                    ypreds, ytrue = [], []
                    self.model.eval()                    

                    for x, y in trainloader:
                        x = x.to(self.device)
                        y = y.to(self.device)

                        # with torch.no_grad():
                        #     outputs, gate_loss, expert_losses, gates, expert_outputs = self.model(x, y)
                        #     old_cls_outs = self.bias_forward(outputs[:task])
                        #     exit()                        

                        outputs, gate_loss, expert_losses, gates, expert_outputs = self.model(x, y)

                        if task > 0:
                            """
                            do bias correction here
                            """
                            bias_outputs = []
                            prev = 0
                            for i, num_class in enumerate(self.previous_task_nums):
                                # print(f"\ti: {i}\tprev: {prev}\tcur: {prev+num_class}")
                                outs = outputs[:, prev:(prev+num_class)]
                                bias_outputs.append(self.model.bias_forward(task, outs))
                                prev += num_class
                            old_cls_outputs = torch.cat(bias_outputs, dim=1)
                            # print(f"\t===> old_cls_outputs.size(): {old_cls_outputs.size()}")

                            new_cls_outputs = self.model.bias_layers[task](outputs[:, prev:])
                            # print(f"\t===> new_cls_outputs.size(): {new_cls_outputs.size()}")
                            pred_all_classes = torch.cat([old_cls_outputs, new_cls_outputs], dim=1)
                            # print(f"\t===> pred_all_classes.size(): {pred_all_classes.size()}")
                            # loss = torch.nn.functional.cross_entropy(pred_all_classes, y)
                            loss = self.criterion(pred_all_classes, y)
                            # print(f"\t===> loss: {loss.item()}")
                            loss += 0.1 * ((self.model.bias_layers[task].beta[0] ** 2) / 2)
                            # print(f"\t===> loss: {loss.item()}\n")
                        else:
                            loss = self.criterion(outputs, y) + gate_loss #+ sum(expert_losses)
                        # loss = self.criterion(outputs, y) + gate_loss + sum(expert_losses)
                        # loss = self.criterion(outputs, y)

                        # if self.previous_model:
                        #     with torch.no_grad():
                        #         prev_outputs, prev_gate_loss, prev_expert_losses, _ = self.previous_model(x, y)
                        #     distillation_loss = self.calculate_loss(outputs, prev_outputs)
                        #     loss = self.lambda_ * distillation_loss + loss #+ prev_gate_loss + sum(prev_expert_losses)
                            # running_dist_loss += distillation_loss * x.size(0)                        

                        total_loss = loss + gate_loss #+ sum(expert_losses)
                        # if task > 0:
                        # print(f"\tLOSS: {loss}\tGATE_LOSS: {gate_loss}\tSUM(EXPERT_LOSSES): {sum(expert_losses)}\tTOTAL_LOSS: {total_loss}")
                        total_loss.backward()
                        # loss.backward()
                        # if the gradient explodes, clip it by value. not sure how often it happens
                        # nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
                        # if task > 0:
                        #     bic_optimiser.step()
                        #     bic_optimiser.zero_grad()
                        # else:
                        # optimiser.step()
                        # optimiser.zero_grad()

                        bias_optimiser.step()
                        bias_optimiser.zero_grad()

                        predicted = torch.argmax(outputs.data, 1) if task == 0 else torch.argmax(pred_all_classes.data, 1)
                        ypreds.extend(predicted.detach().cpu().tolist())
                        ytrue.extend(y.detach().cpu().tolist())

                        running_train_loss += loss.item() * x.size(0)
                        # running_train_loss += total_loss.item()
                        dataset_len += x.size(0)
                    
                    train_loss.append(running_train_loss / dataset_len)
                    # print(f"STEP-2\tEpoch: {epoch+1}/{self.epochs}\tloss: {train_loss[-1]:.4f}\ttrain_accuracy: {(100 * accuracy_score(ytrue, ypreds)):.4f}")      
                    print(f"STEP-2\tEpoch: {epoch+1}/{step2_epochs}\tloss: {train_loss[-1]:.4f}\tstep2_train_accuracy: {(100 * accuracy_score(ytrue, ypreds)):.4f}")      
                self.model.bias_layers[task].alpha.requires_grad = False
                self.model.bias_layers[task].beta.requires_grad = False
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
                            outputs, gate_loss, expert_losses, gates, expert_outputs = self.model(x, y, is_training=False)

                            if task > 0:
                                """
                                do bias correction here
                                """
                                bias_outputs = []
                                prev = 0
                                for i, num_class in enumerate(self.previous_task_nums):
                                    outs = outputs[:, prev:(prev+num_class)]
                                    bias_outputs.append(self.model.bias_forward(task, outs))
                                    prev += num_class
                                old_cls_outputs = torch.cat(bias_outputs, dim=1)
                                new_cls_outputs = self.model.bias_layers[task](outputs[:, prev:])                    
                                pred_all_classes = torch.cat([old_cls_outputs, new_cls_outputs], dim=1)
                                loss = self.criterion(pred_all_classes, y)
                                loss += 0.1 * ((self.model.bias_layers[task].beta[0] ** 2) / 2)
                            else:
                                loss = self.criterion(outputs, y)
                            total_loss = loss + gate_loss #+ sum(expert_losses)                            
                            running_val_loss += total_loss.item() * x.size(0)

                        predicted = torch.argmax(outputs.data, 1) if task == 0 else torch.argmax(pred_all_classes.data, 1)
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
            self.previous_task_nums.append(self.dataset[task]["ncla"])

        return self.train_loss1, self.train_loss2, self.val_loss

    def calculate_loss(self, current_dist, prev_dist):
        # take the current model outputs for old classes
        logits_dist = current_dist[:, :self.seen_cls]
        prev_logits_dist = prev_dist[:, :self.seen_cls]
        log_p = torch.log_softmax(logits_dist / self.temperature, dim=1)        
        q = torch.softmax(prev_logits_dist / self.temperature, dim=1)
        distil_loss = nn.functional.kl_div(log_p, q, reduction="batchmean")

        # soft_pred = torch.softmax(logits_dist/self.temperature, dim=1)
        # soft_target = torch.softmax(prev_logits_dist/self.temperature, dim=1)
        # distil_loss = nn.MSELoss()(soft_pred, soft_target)
        return distil_loss

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
                    outputs, gate_loss, expert_losses, gates, expert_outputs = self.model(x, y)                    
                predicted = torch.argmax(outputs.data, 1)
                ypreds.extend(predicted.detach().cpu().tolist())
                ytrue.extend(y.detach().cpu().tolist())
                
                all_preds.extend(predicted.detach().cpu().tolist())
                all_true.extend(y.detach().cpu().tolist())

            print(f"\tTASK-{task}\tCLASSES: {self.dataset[task]['classes']}\ttest_accuracy: {(100 * accuracy_score(ytrue, ypreds)):.4f}")
        print(f"All accuracy: {(100 * accuracy_score(all_true, all_preds)):.4f}")

    def _get_current_dataloader(self, task, val=False, ignore_replay=False, uniform=False):
        current_dataset = self.dataset[task]
        print(f"CUR_DATASET_CLASSES: {current_dataset['classes']}")
        if ignore_replay:
            x_train, y_train = current_dataset["trn"]["x"], current_dataset["trn"]["y"]
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

    # def bias_forward(self, outputs):
    #     """Utility function --- inspired by https://github.com/sairin1202/BIC"""
    #     bic_outputs = []
    #     for m in range(len(outputs)):
    #         bic_outputs.append(self.bias_layers[m](outputs[m]))
    #     return bic_outputs     

    # def bias_forward(self, task, output):
    #     """Modified version from FACIL"""
    #     return self.bias_layers[task](output)   


if __name__ == "__main__":        
    parser = argparse.ArgumentParser()
    parser.add_argument("-d")
    parser.add_argument("-b", "--batch")
    # parser.add_argument("-c", "--n_class")
    parser.add_argument("-s", "--steps")
    parser.add_argument("-m", "--memory")
    parser.add_argument("-e", "--epochs")
    parser.add_argument("-n", "--n_experts")
    parser.add_argument("-k")
    parser.add_argument("-p", "--pickle")
    parser.add_argument("--wandb_project")

    args = parser.parse_args()

    # d = 0/cifar-100 & 1/imagenet-1000
    d = int(args.d)
    batch = int(args.batch)
    # n_class = int(args.n_class)
    steps = int(args.steps)
    mem_size = int(args.memory)
    epochs = int(args.epochs)
    n_experts = int(args.n_experts)
    k = int(args.k)
    pickle_file = args.pickle 
    wandb_project = args.wandb_project

    train_path = ["cifar100_train_embedding.pt", "imagenet1000_train_embedding.pt"]
    test_path = ["cifar100_test_embedding.pt", "imagenet1000_val_embedding.pt"]
    train_embedding_path = train_path[d]
    val_embedding_path = test_path[d]    

    n_class = 100 if d == 0 else 1000
    lr = 0.001 # 0.001 is the best configuration
    random_replay = RandomReplay(mem_size=mem_size)
    criterion = nn.CrossEntropyLoss()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data, task_cla, class_order = get_data(
        train_embedding_path, 
        None, 
        val_embedding_path=val_embedding_path,
        num_tasks=n_experts,
        expert=True)
    
    met = Metrics()
    trainer = Trainer(
        criterion, data, lr, 
        n_class, class_order, 
        batch_size=batch, 
        epochs=epochs, 
        device=device, 
        replay=random_replay, 
        k=k, 
        metric=met,)

    # for wandb
    config = {
        "minibatch_size": batch,
        "steps": steps,
        "buffer_size": mem_size,
        "epochs": epochs,
        "num_experts": n_experts,
        "k": k,
        "pickle_file": pickle_file,
        "train_embedding_path": train_embedding_path,
        "test_embedding_path": val_embedding_path,
        "n_class": n_class,
        "lr": lr,
        "device": device,
    }

    # if wandb_project:
    #     run = wandb.init(reinit=True, project=wandb_project, config=config)        
    # else:
    #     run = wandb.init(mode="disabled", config=config)

    ##############################################
    """
    run.log()
        - accuracy per task
            - FAA
        - forgetting per task
            - FF
        - running time
        - train loss?
    """
    ##############################################

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
    # if test_embedding_path:
    #     trainer.test_loop()

    losses = {"train_loss1": train_loss1, "train_loss2": train_loss2, "val_loss": val_loss, "metric_acc": met.accuracy, "metric_forget": met.forget}
    pickle.dump(losses, open(pickle_file, "wb"))
