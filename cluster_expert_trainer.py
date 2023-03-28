import argparse
import clusterlib
import copy
import numpy as np
import pickle
import statistics
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import wandb

from collections import Counter
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.models.feature_extraction import create_feature_extractor

from base import BaseDataset, get_data, get_cifar100_coarse
from earlystopping import EarlyStopping
from cluster_expert import DynamicExpert
from mets import Metrics
from replay import RandomReplay
from sklearn.metrics import accuracy_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score, ConfusionMatrixDisplay, confusion_matrix, classification_report

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
        metric=None,
        n_task=5):
    
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
        self.metric = metric
        self.n_task = n_task
    
        self.seen_cls = 0
        self.previous_model = None
        self.previous_cls2idx = None
        self.previous_idx2cls = None
        self.cls2idx = {}
        self.idx2cls = {}
        self.train_loss1 = [] # for losses when training step 1
        self.train_loss2 = [] # for losses when training step 2
        self.val_loss = []
        self.lamb = -1
        self.temperature = 2
        self.previous_task_nums = []
        self.subclass_mapper = None
        # self.buffer_x = None
        # self.buffer_y = None
        self.mem_size = self.replay.mem_size
        self.buffer_classes = None
        
        # now the old self.subclass_mapper is contained in self.rawcls2cluster
        self.rawcls2cluster = {}
        self.cluster2rawcls = {}
        self.finecls2cluster = {}
        self.cluster2finecls = {}
        self.clsgmm = {}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
        self.buffer_x = {}  # {0: [], 1: []}
        self.buffer_y = {}  # {0: [0, 0, 0, 0, 0], 1: [1, 1, 1, 1, 1]}
        self.buffer = {}    # {"x": [], "y": []}

        # if "ordered" in self.dataset:
        #     self.subclass_mapper = self.dataset["ordered"]

        hidden_size = int(1000 / self.n_task)
        class_per_task = int(self.total_cls / self.n_task)
        self.model = DynamicExpert(hidden_size=hidden_size, class_per_task=class_per_task)
    
    def split2subtask(self, x, y, cls2cluster_):
        # split original dataset into sub-tasks dataset
        subtask = {}
        for clust in sorted(set(cls2cluster_.values())):    
            x_, y_ = [], []
            for cls_ in self.cluster2rawcls[clust]:
                x_.extend(x[y == self.cls2idx[cls_]])
                y_.extend(y[y == self.cls2idx[cls_]])
            subtask[clust] = {"x": x_, "y": y_}
        return subtask

    def update_clsgmm(self, clsgmm_):
        self.clsgmm = {**self.clsgmm, **clsgmm_}

    def update_cls2cluster(self, cls2cluster_):
        self.rawcls2cluster = {**self.rawcls2cluster, **cls2cluster_}

    def update_cluster2cls(self, cls2cluster_):
        temp = {}
        for k, v in self.rawcls2cluster.items():
            clust_members = temp.get(v, [])
            clust_members.append(k)
            temp[v] = clust_members
        self.cluster2rawcls = {**self.cluster2rawcls, **temp}

    def update_classmap(self):
        cls_ = []
        for k in sorted(self.cluster2rawcls.keys()):
            members = sorted(self.cluster2rawcls[k])
            cls_.extend(members)

        self.cls2idx = {v: k for k, v in enumerate(cls_)}
        self.idx2cls = {k: v for k, v in enumerate(cls_)}

    def update_finemap(self):
        temp = {}
        for k, v in self.rawcls2cluster.items():
            temp[self.cls2idx[k]] = v
        self.finecls2cluster = temp

        temp = {}
        for k, v in self.cluster2rawcls.items():
            temp[k] = np.vectorize(self.cls2idx.get)(v).tolist()
        self.cluster2finecls = temp

    def get_uniformloader(self, x, y, all=True):
        """
        Take self.mem_size / len(current_seen_classes)
        """
        sample_per_class = 0
        x_t, y_t = None, None
        if all:
            sample_per_class = self.mem_size // len(self.cls2idx)
        else:
            # take old existing class from buffer (if exists)
            # add with current classes
            total_class = 0
            if self.buffer_x:
                total_class += len(self.buffer_y.keys())
            total_class += len(np.unique(y))
            sample_per_class = self.mem_size // total_class

            x_t = np.array(x)
            y_t = np.array(y)


        # if buffer not empty, reduce buffer to current size
        if self.buffer_x:
            # for existing buffer
            for y_, members in self.buffer_x.items():
                self.buffer_x[y_] = self.populate_buffer(members, sample_per_class)
            for y_, members in self.buffer_y.items():
                self.buffer_y[y_] = self.populate_buffer(members, sample_per_class)

            # for new buffer
            class_data = self.dataset_by_class(x, y) if all else self.dataset_by_class(x_t, y_t)
            for label, dataset in class_data.items():
                self.buffer_x[label] = self.populate_buffer(dataset["x"], sample_per_class)
                self.buffer_y[label] = self.populate_buffer(dataset["y"], sample_per_class)
        else:
            class_data = self.dataset_by_class(x, y) if all else self.dataset_by_class(x_t, y_t)
            for label, dataset in class_data.items():
                self.buffer_x[label] = dataset["x"]
                self.buffer_y[label] = dataset["y"]
        
        # create dataloader from self.buffer_x and self.buffer_y
        x_ = [v for k, v in self.buffer_x.items()]
        y_ = [v for k, v in self.buffer_y.items()]

        x_ = [item for sublist in x_ for item in sublist]
        y_ = [item for sublist in y_ for item in sublist]
        
        print(f"\tBUFFER_CLASS: {Counter(y_)}")

        x_ = np.array(x_)
        y_ = np.array(y_)

        dataloader = self._get_dataloader(x_, y_, shuffle=True, batch_size=64)
        return dataloader

    def populate_buffer(self, members, sample_per_class):
        members = np.random.permutation(members)
        members = members[:sample_per_class]
        return members

    def dataset_by_class(self, x, y):
        labels = sorted(np.unique(y))
        class_data = {}
        for label in labels:
            x_ = x[y == label]
            y_ = y[y == label]
            class_data[label] = {"x": x_, "y": y_}
        return class_data

    def draw_heatmap(self, y_true, y_pred, task, filename, title="", big=False):
        print(f"=== drawing heatmap {filename}")
        leg = sorted(np.unique(y_true))
        cmat = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots() if not big else plt.subplots(figsize=(20, 15))
        display = ConfusionMatrixDisplay(cmat, display_labels=leg)

        ax.set_title(title)
        display.plot(ax=ax)
        fig.tight_layout()
        # plt.savefig(f"gate_accuracy_task-{task}.png", dpi=300)
        plt.savefig(f"{filename}.png", dpi=300)
        print("=== finish drawing heatmap")

    def train_loop(self, steps=2):
        val_loaders = []
        cur_task = 0 # pointer for bias correction in Step 2 (only needed for TRAIN_STEP_2_PER_SUBTASK)
        for task in range(self.n_task):            
            # cluster using class mean
            x, y = self.get_numpy_from_dataset(task, type="trn")            
            gmm, cls2cluster_ = clusterlib.class_mean_cluster(x, y, self.cluster2rawcls, seed=42)
            self.update_cls2cluster(cls2cluster_)
            self.update_cluster2cls(cls2cluster_)
            self.update_classmap()
            self.update_finemap()

            # print(f"rawcls2cluster: {self.rawcls2cluster}")
            # print(f"cluster2rawcls: {self.cluster2rawcls}")
            # print(f"cls2idx: {self.cls2idx}")
            # print(f"idx2cls: {self.idx2cls}")
            print(f"finecls2cluster: {self.finecls2cluster}")
            # print(f"cluster2finecls: {self.cluster2finecls}")

            # take the same x, y but with new class mapping
            x, y = self.get_numpy_from_dataset(task, type="trn", transform=True)
            xval, yval = self.get_numpy_from_dataset(task, type="val", transform=True)

            # split original dataset into sub-tasks dataset
            subtask_train = self.split2subtask(x, y, cls2cluster_)
            subtask_val = self.split2subtask(xval, yval, cls2cluster_)

            # create loader for Step 2
            # only when Step 2 is done after all sub-tasks haves been trained
            # otherwise go to TRAIN_STEP_2_PER_SUBTASK block
            # step2loader = self.get_uniformloader(x, y)

            hmap_true, hmap_pred = [], []
            true_labels, expert_output = [], []

            # TRAIN STEP 1
            for subtask_t in sorted(subtask_train.keys()):
                this_x, this_y = subtask_train[subtask_t]["x"], subtask_train[subtask_t]["y"]
                this_xval, this_yval = subtask_val[subtask_t]["x"], subtask_val[subtask_t]["y"]
                
                print(f"SUB-TASK: {subtask_t}\tCLASS: {np.unique(this_y)}\tNUM-CLASS: {len(np.unique(this_y))}")

                this_task_classes = sorted(np.unique(this_y))
                self.model.expand_gmm(this_task_classes)
                self.model.to(self.device)
                
                optimiser = optim.Adam(self.model.parameters(), lr=self.lr)                                                    
                trainloader = self.get_dataloader(this_x, this_y)
                valloader = self.get_dataloader(this_xval, this_yval)
                val_loaders.append(valloader)                

                print("STARTING STEP 1")
                self.model.freeze_previous_experts()
                self.model.set_gate(False)
                
                early_stop = EarlyStopping(verbose=False)
                train_loss = []
                for epoch in range(self.epochs):
                    ypreds, ytrue = [], []
                    self.model.train()

                    running_train_loss = [] # store train_loss per epoch
                    dataset_len = 0
                    pred_correct = 0.0

                    for inputs, labels in trainloader:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        
                        # gate_outputs will be None for train_step=1
                        # original_expert_outputs will be ignored for train_step=1
                        outputs, gate_outputs, original_expert_outputs = self.model(inputs, subtask_t, train_step=1)    
                                                
                        # print(outputs)
                        # print(labels)
                        # exit()
                        loss = self.criterion(outputs, labels)
                        loss.backward()
                        optimiser.step()
                        optimiser.zero_grad()
                                    
                        predicted = torch.argmax(outputs, 1)
                        pred_correct += predicted.eq(labels).cpu().sum().float()

                        running_train_loss.append(loss.item())
                        dataset_len += inputs.size(0) #+ inputs.size(0)
                    
                    early_stop(loss, self.model)
                    train_loss.append(np.average(running_train_loss))
                    if (epoch + 1) % 10 == 0:
                        print(f"STEP-1\tEpoch: {epoch+1}/{self.epochs}\tloss: {train_loss[-1]:.4f}\tstep1_train_accuracy: {(100 * pred_correct / dataset_len):.4f}")
                    if early_stop.early_stop:
                        print(f"Early stopping. Exit epoch {epoch+1}")
                        break
                self.train_loss1.append(train_loss)
                print("FINISH STEP 1\n")

                # TRAIN THE GATE
                # unindent 1 if Step 2 is done after all subtasks have been trained

                # TRAIN_STEP_2_PER_SUBTASK
                step2loader = self.get_uniformloader(this_x, this_y, all=False)
                if steps == 2:
                    print("STARTING STEP 2")
                    self.model.freeze_all_experts()
                    self.model.set_gate(True)

                    train_loss = [] # store train_loss per epoch
                    gate_loss_ = [] # store gate_loss per epoch

                    gate_optimiser = optim.Adam(self.model.parameters(), lr=self.lr)
                    early_stop = EarlyStopping(verbose=False)
                    for epoch in range(self.epochs):
                        running_train_loss, running_gate_loss = [], []
                        dataset_len, gate_correct, pred_correct = 0, 0.0, 0.0
                        ypreds, ytrue = [], []
                        
                        this_epoch_gate_true, this_epoch_gate_pred = [], []
                        this_epoch_labels, this_epoch_expert_pred = [], []

                        total_sample = 0
                        for i, (images, labels) in enumerate(step2loader):
                            images = images.to(self.device)
                            labels = labels.to(self.device)
                            
                            outputs, gate_outputs, original_expert_outputs = self.model(images, train_step=2)

                            for out in original_expert_outputs:
                                print(out.size())
                                total_sample += out.size(0)

                            if cur_task > 0:
                            # RUN BIAS CORRECTION FOR ALL EXPERTS
                                bias_outputs = []
                                prev = 0
                                # for i, clust_i in enumerate(sorted(self.cluster2finecls.keys())):
                                for clust_i in range(cur_task):
                                    num_class = len(self.cluster2finecls[clust_i])
                                    outs = outputs[:, prev:(prev+num_class)]
                                    bias_outputs.append(self.model.bias_forward(clust_i, outs))
                                    prev += num_class
                                old_cls_outputs = torch.cat(bias_outputs, dim=1)
                                new_cls_outputs = self.model.bias_forward(clust_i, outputs[:, prev:])  # prev should point to the last subtask
                                pred_all_classes = torch.cat([old_cls_outputs, new_cls_outputs], dim=1)
                                loss = self.criterion(pred_all_classes, labels)
                                loss += 0.1 * ((self.model.bias_layers[clust_i].beta[0] ** 2) / 2)
                            else:
                                if len(outputs.size()) == 1:
                                    outputs = outputs.view(-1, 1)
                                loss = self.criterion(outputs, labels)
                            
                            running_train_loss.append(loss.item())
                            
                            gate_labels = torch.tensor(np.vectorize(self.finecls2cluster.get)(labels.cpu().numpy())).type(torch.LongTensor).to(self.device)
                            gate_loss = self.criterion(gate_outputs, gate_labels)

                            running_gate_loss.append(gate_loss.item())
                            loss += gate_loss
                            gate_preds = torch.argmax(gate_outputs.data, 1)
                            gate_correct += gate_preds.eq(gate_labels).cpu().sum().float()

                            # book keeping for gate prediction
                            # this_epoch_gate_true.extend(gate_labels.cpu().numpy().tolist())
                            # this_epoch_gate_pred.extend(gate_preds.cpu().numpy().tolist())
                            # print(f"\t\t\tFine labels: {labels.cpu().numpy()}\n\t\t\tGate labels: {gate_labels.cpu().numpy()}\n\t\t\tGate preds: {gate_preds.cpu().numpy()}\n\t\t\tGate correct: {len(gate_labels)}/{gate_preds.eq(gate_labels).cpu().sum().int()}\n")

                            # book keeping for expert output
                            # this_epoch_expert_pred.extend(torch.argmax(outputs.data, 1).cpu().numpy().tolist())
                            # this_epoch_labels.extend(labels.cpu().numpy().tolist())                            

                            loss.backward()
                            gate_optimiser.step()
                            gate_optimiser.zero_grad()

                            # predicted = torch.argmax(outputs.data, 1) if clust_i == 0 else torch.argmax(pred_all_classes, 1)
                            # predicted = torch.argmax(outputs.data, 1)

                            # predicted = torch.argmax(pred_all_classes, 1)
                            # TRAIN_STEP_2_PER_SUBTASK
                            predicted = torch.argmax(outputs.data, 1) if cur_task == 0 else torch.argmax(pred_all_classes, 1)
                            
                            # print(predicted)
                            # print(labels)
                            # print()
                            pred_correct += predicted.eq(labels).cpu().sum().float()

                            ypreds.extend(predicted.detach().cpu().tolist())
                            ytrue.extend(labels.detach().cpu().tolist())
                            
                            dataset_len += images.size(0)                    

                        # hmap_true.append(this_epoch_gate_true)
                        # hmap_pred.append(this_epoch_gate_pred)

                        expert_output.append(this_epoch_expert_pred)
                        true_labels.append(this_epoch_labels)

                        early_stop(loss, self.model)

                        train_loss.append(np.average(running_train_loss))
                        gate_loss_.append(np.average(running_gate_loss))
                        # if (epoch + 1) % 10 == 0:                                
                        print(f"STEP-2\tEpoch: {epoch+1}/{self.epochs}\tclassification_loss: {train_loss[-1]:.4f}\tgate_loss: {gate_loss_[-1]:.4f}\tstep2_classification_accuracy: {(100 * pred_correct.item() / dataset_len):.4f}\tstep_2_gate_accuracy: {100 * (gate_correct / dataset_len):.4f}")
                            # print(f"STEP-2\tEpoch: {epoch+1}/{self.epochs}\tclassification_loss: {train_loss[-1]:.4f}\tstep2_classification_accuracy: {(100 * pred_correct.item() / dataset_len):.4f}")
                        if early_stop.early_stop:
                            print(f"Early stopping. Exit epoch {epoch+1}")
                            break
                    print("FINISH STEP 2\n")                    
                self.model.unfreeze_all()                
                cur_task += 1

            # self.draw_heatmap(hmap_true[-1], hmap_pred[-1], task, f"gate_accuracy_task-{task}", title=f"Gate Accuracy Task-{task}")            
            # self.draw_heatmap(true_labels[-1], expert_output[-1], task, f"expert_accuracy_task-{task}", title=f"Expert Accuracy Task-{task}", big=True)

            # print(f"\n===TRUE LABELS VS. EXPERT OUTPUT TASK-{task}===")
            # print(classification_report(y_true=true_labels[-1], y_pred=expert_output[-1]))
            # print("================================================\n")

            exit()

            """
            CLOSE HERE
            if val_loaders:
                self.model.eval()
                val_loss_ = []                
                for subtask, valloader in enumerate(val_loaders):                    
                    running_val_loss = []
                    dataset_len = 0
                    gate_correct = 0
                    pred_correct = 0.0
                    ypreds, ytrue = [], []
                    for i, (images, labels) in enumerate(valloader):                        
                        images = images.to(self.device)
                        labels = labels.to(self.device)

                        with torch.no_grad():
                            outputs, gate_outputs = self.model(images, train_step=2)

                        bias_outputs = []
                        prev = 0
                        # for i, clust_i in enumerate(sorted(self.cluster2rawcls.keys())):
                        for clust_i in range(cur_task):
                            num_class = len(self.cluster2finecls[clust_i])
                            outs = outputs[:, prev:(prev+num_class)]
                            bias_outputs.append(self.model.bias_forward(clust_i, outs))
                            prev += num_class
                        old_cls_outputs = torch.cat(bias_outputs, dim=1)
                        new_cls_outputs = self.model.bias_forward(clust_i, outputs[:, prev:])  # prev should point to the last task
                        pred_all_classes = torch.cat([old_cls_outputs, new_cls_outputs], dim=1)
                        loss = self.criterion(pred_all_classes, labels)
                        loss += 0.1 * ((self.model.bias_layers[clust_i].beta[0] ** 2) / 2)

                        gate_labels = torch.tensor(np.vectorize(self.finecls2cluster.get)(labels.cpu().numpy())).type(torch.LongTensor).to(self.device)
                        gate_loss = self.criterion(gate_outputs, gate_labels)

                        loss += gate_loss
                        gate_preds = torch.argmax(gate_outputs.data, 1)
                        gate_correct += gate_preds.eq(gate_labels).cpu().sum().float()                                

                        running_val_loss.append(loss.item())
                        dataset_len += images.size(0)

                        predicted = torch.argmax(pred_all_classes, 1)
                        pred_correct += predicted.eq(labels).cpu().sum().float()
                        ytrue.extend(labels.detach().cpu().tolist())

                        # print(f"\t\tpred_correct: {pred_correct}\tdataset_len: {dataset_len}")

                    val_loss_.append(np.average(running_val_loss))
                    task_accuracy = 100 * (pred_correct / dataset_len)
                    print(f"\tSubtask-{subtask}\tval_loss: {val_loss_[-1]:.4f}\tval_accuracy: {task_accuracy:.4f}\tgate_accuracy: {100 * (gate_correct / dataset_len):.4f}")
                print()
                #     if self.metric:
                #         self.metric.add_accuracy(subtask, task_accuracy)
                # self.val_loss.append(val_loss_)

                # if self.metric:
                #     self.metric.add_forgetting(subtask)
            UNTIL HERE
            """

            # self.seen_cls += new_cls
            # self.previous_task_nums.append(self.dataset[task]["ncla"])

        print("done for all tasks")
        exit()
        return self.train_loss1, self.train_loss2, self.val_loss, self.model

    def test_loop(self):
        self.model.to(self.device)
        self.model.eval()
        print("CALCULATING TEST LOSS PER TASK")

        all_preds, all_true = [], []   
        for task in range(self.n_task):
            ypreds, ytrue = [], []
            x_test, y_test = self.dataset[task]["tst"]["x"], self.dataset[task]["tst"]["y"]
            testloader = self._get_dataloader(x_test, y_test)
            
            for x, y in testloader:
                x = x.to(self.device)
                y = y.to(self.device)

                with torch.no_grad():
                    outputs, gate_outputs = self.model(x)
                    if self.model.bias_layers:
                        bias_outputs = []
                        prev = 0
                        for i, num_class in enumerate(self.previous_task_nums):
                            outs = outputs[:, prev:(prev+num_class)]
                            bias_outputs.append(self.model.bias_forward(task, outs))
                            prev += num_class
                        pred_all_classes = torch.cat(bias_outputs, dim=1)
                
                predicted = torch.argmax(outputs.data, 1) if not self.model.bias_layers else torch.argmax(pred_all_classes.data, 1)
                ypreds.extend(predicted.detach().cpu().tolist())
                ytrue.extend(y.detach().cpu().tolist())
                
                all_preds.extend(predicted.detach().cpu().tolist())
                all_true.extend(y.detach().cpu().tolist())

            print(f"\tTASK-{task}\tCLASSES: {self.dataset[task]['classes']}\ttest_accuracy: {(100 * accuracy_score(ytrue, ypreds)):.4f}")
        print(f"All accuracy: {(100 * accuracy_score(all_true, all_preds)):.4f}")

    def get_numpy_from_dataset(self, task, type="trn", transform=False):
        current_dataset = self.dataset[task]
        # if type == "val" and len(current_dataset["val"]["x"])        
        x = torch.vstack(current_dataset[type]["x"]).numpy()
        y = torch.tensor(current_dataset[type]["y"]).numpy()
        if transform:
            y = np.vectorize(self.cls2idx.get)(y)
        return x, y

    def get_dataloader(self, x, y):
        dataloader = self._get_dataloader(x, y, shuffle=True)
        return dataloader

    def update_buffer(self, task, uniform=True):
        current_dataset = self.dataset[task]
        self.replay.update_buffer(current_dataset, uniform=True)

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
            
        batch_size = 64 if uniform else None
        trainloader = self._get_dataloader(x_train, y_train, shuffle=True, batch_size=batch_size)

        valloader = None
        if len(current_dataset["val"]["x"]) != 0 and val:
            x_val, y_val = current_dataset["val"]["x"], current_dataset["val"]["y"]
            valloader = self._get_dataloader(x_val, y_val)

        class_counter = Counter(y_train)
        print("CLASS_COUNTER:", class_counter)

        return trainloader, valloader

    def _get_dataloader(self, x, y, shuffle=False, batch_size=None):
        # _dataset = BaseDataset(x, y, self.transform, self.cls2idx)
        _dataset = BaseDataset(x, y, self.transform) # no need cls2idx as it's already mapped
        batch_size = batch_size if batch_size else self.batch_size
        _dataloader = DataLoader(_dataset, batch_size=batch_size, shuffle=shuffle)
        return _dataloader

    def reduce_learning_rate(self, optimiser):        
        for param_group in optimiser.param_groups:
            old_lr = param_group["lr"]
        for param_group in optimiser.param_groups:
            param_group["lr"] = param_group["lr"] / 10
            new_lr = param_group["lr"]
        print(f"Reduce learning_rate from {old_lr} to {new_lr}")

    def print_params(self):
        for name, param in self.model.named_parameters():
            print(name, param.requires_grad)


if __name__ == "__main__":        
    parser = argparse.ArgumentParser()
    parser.add_argument("-d")
    parser.add_argument("-b", "--batch")
    parser.add_argument("-s", "--steps")
    parser.add_argument("-m", "--memory")
    parser.add_argument("-e", "--epochs")
    parser.add_argument("-n", "--n_experts")
    # parser.add_argument("-k")
    parser.add_argument("-p", "--pickle")
    parser.add_argument("--wandb_project")

    args = parser.parse_args()

    # d = 0/cifar-100 & 1/imagenet-1000 & 2/cifar100-coarse
    d = int(args.d)
    batch = int(args.batch)
    # n_class = int(args.n_class)
    steps = int(args.steps)
    mem_size = int(args.memory)
    epochs = int(args.epochs)
    n_experts = int(args.n_experts)
    # k = int(args.k)
    pickle_file = args.pickle 
    wandb_project = args.wandb_project

    train_path = ["cifar100_train_embedding.pt", "imagenet1000_train_embedding.pt", "cifar100_coarse_train_embedding_nn.pt"]
    test_path = ["cifar100_test_embedding.pt", "imagenet1000_val_embedding.pt", "cifar100_coarse_test_embedding_nn.pt"]
    train_embedding_path = train_path[d]
    val_embedding_path = None
    test_embedding_path = test_path[d]

    n_class = 1000 if d == 1 else 100
    lr = 0.01 # 0.001 is the best configuration
    random_replay = RandomReplay(mem_size=mem_size)
    criterion = nn.CrossEntropyLoss()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if d != 2:
        data, task_cla, class_order = get_data(
            train_embedding_path, 
            test_embedding_path, 
            validation=0.2,
            num_tasks=n_experts,
            expert=True,
            seed=42)
    else:
        test_embedding_path = test_path[d]
        # data, task_cla, class_order = get_cifar100_coarse(train_embedding_path, test_embedding_path, None, validation=0.2)
        data, task_cla, class_order = get_cifar100_coarse(train_embedding_path, test_embedding_path, None)
        
    met = Metrics()
    trainer = Trainer(
        criterion, data, lr, 
        n_class, class_order, 
        batch_size=batch, 
        epochs=epochs, 
        device=device, 
        replay=random_replay, 
        metric=met,
        n_task=n_experts)

    # for wandb
    config = {
        "minibatch_size": batch,
        "steps": steps,
        "buffer_size": mem_size,
        "epochs": epochs,
        "num_experts": n_experts,
        # "k": k,
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
    train_loss1, train_loss2, val_loss, model = trainer.train_loop(steps=steps)
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
    pickle.dump(losses, open(pickle_file, "wb"))

    # model_path = pickle_file.split(".")[0]
    # torch.save(model, f"{model_path}.pth")
    
    # to_save = {"data": data, "task_cla": task_cla, "class_order": class_order}
    # pickle.dump(to_save, open(f"{model_path}.dat", "wb"))


# l = nn.Linear(2, 4)
# w1 = weight_norm(l, name="weight")
# w1.weight_g

# python cluster_expert_trainer.py -d 0 -b 256 -s 2 -m 500 -e 100 -n 5 -p "z.pkl"