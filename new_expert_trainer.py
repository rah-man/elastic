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

from base import BaseDataset, get_data, get_cifar100_coarse
from new_expert import DynamicExpert
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
        self.lambda_= 1.6
        self.temperature = 2
        self.previous_task_nums = []
        self.subclass_mapper = None

        if "ordered" in self.dataset:
            self.subclass_mapper = self.dataset["ordered"]

        # hidden_size = int(self.total_cls / (len(self.dataset)-1))
        hidden_size = int(1000 / self.n_task)
        self.model = DynamicExpert(hidden_size=hidden_size)

    def update_classmap(self, new_cls):
        cls_ = list(self.cls2idx.keys())
        cls_.extend(new_cls)
        self.cls2idx = {v: k for k, v in enumerate(cls_)}
        self.idx2cls = {k: v for k, v in enumerate(cls_)}

    def train_loop(self, steps=2):        
        val_loaders = []        
        final_train_batch = []
        for task in range(self.n_task):
            # update the model for new classes in each iteration
            new_cls = len(self.dataset[task]["classes"])
            self.update_classmap(self.dataset[task]["classes"])

            self.model.expand_expert(self.seen_cls, new_cls)

            print("=====GATE WEIGHT BEGIN=====")
            self.model.calculate_gate_norm()
            print("==========================")

            if task > 0:
                self.previous_model = copy.deepcopy(self.model)
                self.previous_model.to(device)
                self.previous_model.eval()            

            self.model.to(self.device)            
            # optimiser = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.8)
            optimiser = optim.Adam(self.model.parameters(), lr=self.lr, )
            
            if steps == 2:
                trainloader, valloader = self._get_current_dataloader(task, val=True, ignore_replay=True) 
            elif steps == 1:
                trainloader, valloader = self._get_current_dataloader(task, val=True) 
            val_loaders.append(valloader)

            # ADD EARLY STOPPING FOR STEP 1 BUT NOT FOR STEP 2
            print("STARTING STEP 1")
            self.model.freeze_previous()   
            self.model.set_gate(False)             
            train_loss = []

            batch_store = []
            for epoch in range(self.epochs):
                ypreds, ytrue = [], []
                self.model.train()

                # running_train_loss = 0.0
                running_train_loss = []
                dataset_len = 0

                for x, y in trainloader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    outputs, gate_outputs = self.model(x, y, task, train_step=1)
                    batch_store.append((x, y))

                    loss = self.criterion(outputs, y)
                    loss.backward()
                    optimiser.step()
                    optimiser.zero_grad()
                                  
                    predicted = torch.argmax(outputs.data, 1)
                    ypreds.extend(predicted.detach().cpu().tolist())
                    ytrue.extend(y.detach().cpu().tolist())

                    # running_train_loss += loss.item() * x.size(0)
                    # dataset_len += x.size(0)
                    running_train_loss.append(loss.item())
                
                # scheduler.step()
                # train_loss.append(running_train_loss / dataset_len)            
                train_loss.append(np.average(running_train_loss))
                print(f"STEP-1\tEpoch: {epoch+1}/{self.epochs}\tloss: {train_loss[-1]:.4f}\tstep1_train_accuracy: {(100 * accuracy_score(ytrue, ypreds)):.4f}")

            self.train_loss1.append(train_loss)

            print("FINISH STEP 1")

            if steps == 2:
                print("STARTING STEP 2")
                self.model.freeze_all()
                self.model.set_gate(True)

                trainloader, _ = self._get_current_dataloader(task, uniform=True)
                train_loss = [] # store train_loss per epoch
                gate_loss_ = [] # store gate_loss per epoch

                # gate_optimiser = optim.SGD(self.model.parameters(), lr=self.lr)
                gate_optimiser = optim.Adam(self.model.parameters(), lr=self.lr)
                for epoch in range(self.epochs):
                    # running_train_loss, running_gate_loss = 0.0, 0.0
                    running_train_loss, running_gate_loss = [], []
                    dataset_len, gate_correct = 0, 0
                    ypreds, ytrue = [], []

                    for i, (x, y) in enumerate(trainloader):
                        x = x.to(self.device)
                        y = y.to(self.device)
                        
                        # THERE MIGHT BE A MISTAKE HERE REGARDING task=task
                        outputs, gate_outputs = self.model(x, y, task=task)

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
                            # print(f"\tpred_all_loss: {loss}")
                            loss += 0.1 * ((self.model.bias_layers[task].beta[0] ** 2) / 2)
                            # print(f"\tbias_loss: {loss}\n")
                        else:
                            if len(outputs.size()) == 1:
                                outputs = outputs.view(-1, 1)
                            loss = self.criterion(outputs, y)

                        # running_train_loss += loss.item() * x.size(0)
                        running_train_loss.append(loss.item())
                        
                        # CALCULATE GATE LOSS BY MAPPING Y TO GATE
                        if self.subclass_mapper:
                            gate_labels = torch.tensor(np.vectorize(self.subclass_mapper.get)(y.cpu().numpy())).type(torch.LongTensor).to(self.device)
                            gate_loss = self.criterion(gate_outputs, gate_labels)
                            # print(f"\tgate_loss: {gate_loss}")

                            # running_gate_loss += gate_loss.item() * x.size(0)
                            running_gate_loss.append(gate_loss.item())
                            loss += gate_loss

                            gate_preds = torch.argmax(gate_outputs.data, 1)
                            gate_correct += (gate_preds == gate_labels).sum().item()

                            # if task == 4:# and epoch == self.epochs-1 and i == len(trainloader) - 1:
                            #     print(f"\ty: {y}")
                            #     print(f"\tGATE_LABELS: {gate_labels}")
                            #     print(f"\tGATE_OUTPUTS: {gate_outputs}")
                            #     print(f"\tGATE_LOSS: {gate_loss}")
                            #     print(f"\tGATE_PREDS: {gate_preds}")
                            #     print(f"\tGATE_CORRECT: {gate_correct}")                                
                            #     print()                            

                        loss.backward()
                        gate_optimiser.step()
                        gate_optimiser.zero_grad()

                        predicted = torch.argmax(outputs.data, 1)
                        ypreds.extend(predicted.detach().cpu().tolist())
                        ytrue.extend(y.detach().cpu().tolist())
                        
                        dataset_len += x.size(0)
                    
                    # train_loss.append(running_train_loss / dataset_len)
                    # gate_loss_.append(running_gate_loss / dataset_len)
                    train_loss.append(np.average(running_train_loss))
                    gate_loss_.append(np.average(running_gate_loss))
                    print(f"STEP-2\tEpoch: {epoch+1}/{self.epochs}\tclassification_loss: {train_loss[-1]:.4f}\tgate_loss: {gate_loss_[-1]:.4f}\tstep2_train_accuracy: {(100 * accuracy_score(ytrue, ypreds)):.4f}\tstep_2_gate_accuracy: {100 * (gate_correct / dataset_len):.4f}")
                print("FINISH STEP 2")
            self.model.unfreeze_all()

            print("=====GATE WEIGHT END=====")
            self.model.calculate_gate_norm()
            print("========================")        

            if val_loaders:
                self.model.eval()                                
                val_loss_ = []                
                for nt, valloader in enumerate(val_loaders):
                    # running_val_loss = 0.0
                    running_val_loss = []
                    dataset_len = 0
                    ypreds, ytrue = [], []
                    for i, (x, y) in enumerate(valloader):
                        gate_correct = 0
                        x = x.to(self.device)
                        y = y.to(self.device)

                        with torch.no_grad():
                            outputs, gate_outputs = self.model(x, y, task=nt)

                            if nt > 0:
                                bias_outputs = []
                                prev = 0
                                for i, num_class in enumerate(self.previous_task_nums):
                                    outs = outputs[:, prev:(prev+num_class)]
                                    bias_outputs.append(self.model.bias_forward(nt, outs))
                                    prev += num_class
                                old_cls_outputs = torch.cat(bias_outputs, dim=1)
                                new_cls_outputs = self.model.bias_layers[nt](outputs[:, prev:])                    
                                pred_all_classes = torch.cat([old_cls_outputs, new_cls_outputs], dim=1)
                                loss = self.criterion(pred_all_classes, y)
                                loss += 0.1 * ((self.model.bias_layers[nt].beta[0] ** 2) / 2)
                            else:
                                if len(outputs.size()) == 1:
                                    outputs = outputs.view(-1, 1)                                
                                loss = self.criterion(outputs, y)

                            if self.subclass_mapper:
                                one_hot_class = gate_outputs.size(1)
                                gate_labels = torch.tensor(np.vectorize(self.subclass_mapper.get)(y.cpu().numpy())).type(torch.LongTensor).to(self.device)
                                gate_loss = self.criterion(gate_outputs, gate_labels)
                                # print(f"\tgate_loss: {gate_loss}")
                                loss += gate_loss

                                # if i == len(valloader):
                                gate_preds = torch.argmax(gate_outputs.data, 1)
                                gate_correct += (gate_preds == gate_labels).sum().item()

                                # print(f"\ty: {y}")
                                # print(f"\tGATE_LABELS: {gate_labels}")
                                # print(f"\tGATE_OUTPUTS: {gate_outputs}")
                                # print(f"\tGATE_LOSS: {gate_loss}")
                                # print(f"\tGATE_PREDS: {gate_preds}")
                                # print(f"\tGATE_CORRECT: {gate_correct}")
                                # print()            

                            # running_val_loss += loss.item() * x.size(0)
                            running_val_loss.append(loss.item())

                        predicted = torch.argmax(outputs.data, 1)
                        ypreds.extend(predicted.detach().cpu().tolist())
                        ytrue.extend(y.detach().cpu().tolist())
                        dataset_len += x.size(0)

                    
                    # val_loss_.append(running_val_loss / dataset_len)            
                    val_loss_.append(np.average(running_val_loss))
                    task_accuracy = 100 * accuracy_score(ytrue, ypreds)
                    print(f"\tTask-{nt} val_loss: {val_loss_[-1]:.4f}\tval_accuracy: {task_accuracy:.4f}")
                    if self.metric:
                        self.metric.add_accuracy(task, task_accuracy)
                self.val_loss.append(val_loss_)

                final_train_batch.append(batch_store[0])
            if self.metric:
                self.metric.add_forgetting(task)
            print()

            self.seen_cls += new_cls
            self.previous_task_nums.append(self.dataset[task]["ncla"])
        
        for nt, (x, y) in enumerate(final_train_batch):
            gate_correct = 0
            x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                outputs, gate_outputs = self.model(x, y, task=nt)
                if nt > 0:
                    bias_outputs = []
                    prev = 0
                    for i, num_class in enumerate(self.previous_task_nums):
                        outs = outputs[:, prev:(prev+num_class)]
                        bias_outputs.append(self.model.bias_forward(nt, outs))
                        prev += num_class
                    old_cls_outputs = torch.cat(bias_outputs, dim=1)
                    new_cls_outputs = self.model.bias_layers[nt](outputs[:, prev:])
                    pred_all_classes = torch.cat([old_cls_outputs, new_cls_outputs], dim=1)
                    loss = self.criterion(pred_all_classes, y)
                    loss += 0.1 * ((self.model.bias_layers[nt].beta[0] ** 2) / 2)
                else:
                    if len(outputs.size()) == 1:
                        outputs = outputs.view(-1, 1)                                
                    loss = self.criterion(outputs, y)

                if self.subclass_mapper:
                    gate_labels = torch.tensor(np.vectorize(self.subclass_mapper.get)(y.cpu().numpy())).type(torch.LongTensor).to(self.device)
                    gate_loss = self.criterion(gate_outputs, gate_labels)
                    loss += gate_loss

                    gate_preds = torch.argmax(gate_outputs.data, 1)
                    gate_correct += (gate_preds == gate_labels).sum().item()

                    print(f"\ty: {y}")
                    print(f"\tGATE_LABELS: {gate_labels}")
                    print(f"\tGATE_OUTPUTS: {gate_outputs}")
                    print(f"\tGATE_LOSS: {gate_loss}")
                    print(f"\tGATE_PREDS: {gate_preds}")
                    print(f"\tGATE_CORRECT: {gate_correct}")
                    print()            

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
                    outputs, gate_outputs = self.model(x, y)
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
        _dataset = BaseDataset(x, y, self.transform, self.cls2idx)
        batch_size = batch_size if batch_size else self.batch_size
        _dataloader = DataLoader(_dataset, batch_size=self.batch_size, shuffle=shuffle)
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
    val_embedding_path = test_path[d]    
    test_embedding_path = None

    n_class = 1000 if d == 1 else 100
    lr = 0.01 # 0.001 is the best configuration
    random_replay = RandomReplay(mem_size=mem_size)
    criterion = nn.CrossEntropyLoss()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if d != 2:
        data, task_cla, class_order = get_data(
            train_embedding_path, 
            None, 
            val_embedding_path=val_embedding_path,
            num_tasks=n_experts,
            expert=True)
    else:
        test_embedding_path = test_path[d]
        data, task_cla, class_order = get_cifar100_coarse(train_embedding_path, test_embedding_path, None, validation=0.2)
        
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