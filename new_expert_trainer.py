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
from earlystopping import EarlyStopping
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
        self.lamb = -1
        self.temperature = 2
        self.previous_task_nums = []
        self.subclass_mapper = None

        if "ordered" in self.dataset:
            self.subclass_mapper = self.dataset["ordered"]

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
            # print("\tBEGIN_CLS2IDX:", self.cls2idx)
            early_stop = EarlyStopping(verbose=True)
            
            # update the model for new classes in each iteration
            new_cls = len(self.dataset[task]["classes"])
            self.update_classmap(self.dataset[task]["classes"])

            self.model.expand_expert(self.seen_cls, new_cls)

            # print("=====GATE WEIGHT BEGIN=====")
            # self.model.calculate_gate_norm()
            # print("==========================")

            if task > 0:
                self.previous_model = copy.deepcopy(self.model)
                self.previous_model.to(device)
                self.previous_model.eval()            

            self.model.to(self.device)            
            optimiser = optim.Adam(self.model.parameters(), lr=self.lr, )
            
            if steps == 2:
                trainloader, valloader = self._get_current_dataloader(task, val=True, ignore_replay=True) 
            elif steps == 1:
                trainloader, valloader = self._get_current_dataloader(task, val=True) 
            val_loaders.append(valloader)

            print("STARTING STEP 1")
            self.model.freeze_previous_experts()
            self.model.set_gate(False)             
            train_loss = []

            for epoch in range(self.epochs):
                ypreds, ytrue = [], []
                self.model.train()

                running_train_loss = [] # store train_loss per epoch
                dataset_len = 0
                pred_correct = 0.0

                for x, y in trainloader:
                    x = x.to(self.device)
                    y = y.to(self.device)

                    # inputs, targets_a, targets_b, lam = self.mixup_data(x, y)

                    outputs, gate_outputs = self.model(x, task, train_step=1)    # gate_outputs will be None for train_step=1
                    # outputs_mix, gate_outputs_mix = self.model(inputs, task, train_step=1)

                    loss = self.criterion(outputs, y)
                    # loss += self.mixup_criterion(self.criterion, outputs_mix, targets_a, targets_b, lam)

                    loss.backward()
                    optimiser.step()
                    optimiser.zero_grad()
                                  
                    predicted = torch.argmax(outputs, 1)
                    pred_correct += predicted.eq(y).cpu().sum().float()
                    # predicted = torch.argmax(outputs_mix, 1)
                    # pred_correct += (lam * predicted.eq(targets_a).cpu().sum().float() + (1 - lam) * predicted.eq(targets_b).cpu().sum().float())

                    running_train_loss.append(loss.item())

                    dataset_len += x.size(0) #+ inputs.size(0)
                
                train_loss.append(np.average(running_train_loss))
                if (epoch + 1) % 10 == 0:
                    print(f"STEP-1\tEpoch: {epoch+1}/{self.epochs}\tloss: {train_loss[-1]:.4f}\tstep1_train_accuracy: {(100 * pred_correct / dataset_len):.4f}")

            self.train_loss1.append(train_loss)

            print("FINISH STEP 1")

            # TRAIN THE GATE
            if steps == 2:
                print("STARTING STEP 2")
                self.model.freeze_all_experts()
                self.model.set_gate(True)

                trainloader, _ = self._get_current_dataloader(task, uniform=True)
                train_loss = [] # store train_loss per epoch
                gate_loss_ = [] # store gate_loss per epoch

                gate_optimiser = optim.Adam(self.model.parameters(), lr=self.lr)
                # self.update_buffer(task, uniform=True)  # do it here once                
                # for epoch in range(self.epochs):
                for epoch in range(200):
                    running_train_loss, running_gate_loss = [], []
                    dataset_len, gate_correct, pred_correct = 0, 0.0, 0.0
                    ypreds, ytrue = [], []
                    
                    for i, (x, y) in enumerate(trainloader):
                        x = x.to(self.device)
                        y = y.to(self.device)
                        inputs, targets_a, targets_b, lam = self.mixup_data(x, y)
                        
                        # outputs_ori, gate_outputs_ori = self.model(x, train_step=2)
                        outputs, gate_outputs = self.model(inputs, train_step=2)

                        if task > 0:
                            bias_outputs = []
                            # bias_outputs_ori = []
                            prev = 0
                            for i, num_class in enumerate(self.previous_task_nums):
                                outs = outputs[:, prev:(prev+num_class)]
                                # outs_ori = outputs_ori[:, prev:(prev+num_class)]

                                bias_outputs.append(self.model.bias_forward(task, outs))
                                # bias_outputs_ori.append(self.model.bias_forward(task, outs_ori))

                                prev += num_class

                            old_cls_outputs = torch.cat(bias_outputs, dim=1)
                            # old_cls_outputs_ori = torch.cat(bias_outputs_ori, dim=1)

                            new_cls_outputs = self.model.bias_forward(task, outputs[:, prev:])  # prev should point to the last task
                            # new_cls_outputs_ori = self.model.bias_forward(task, outputs_ori[:, prev:])  # prev should point to the last task

                            pred_all_classes = torch.cat([old_cls_outputs, new_cls_outputs], dim=1)
                            # pred_all_classes_ori = torch.cat([old_cls_outputs_ori, new_cls_outputs_ori], dim=1)
                            # loss = self.criterion(pred_all_classes, y)
                            # loss = self.mixup_criterion(self.criterion, pred_all_classes, y_ori, y_mix, lam)
                            loss = self.mixup_criterion(self.criterion, pred_all_classes, targets_a, targets_b, lam)
                            # loss += self.criterion(pred_all_classes_ori, y)
                            loss += 0.1 * ((self.model.bias_layers[task].beta[0] ** 2) / 2)
                        else:
                            if len(outputs.size()) == 1:
                                outputs = outputs.view(-1, 1)
                                # outputs_ori = outputs_ori.view(-1, 1)
                            # loss = self.criterion(outputs, y)
                            # loss = self.mixup_criterion(self.criterion, outputs, y_ori, y_mix, lam)
                            loss = self.mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
                            # loss += self.criterion(outputs_ori, y)
                        
                        running_train_loss.append(loss.item())

                        if self.subclass_mapper:
                            # ori_labels = torch.tensor(np.vectorize(self.subclass_mapper.get)(y.cpu().numpy())).type(torch.LongTensor).to(self.device)
                            gate_labels = torch.tensor(np.vectorize(self.subclass_mapper.get)(targets_a.cpu().numpy())).type(torch.LongTensor).to(self.device)
                            mix_labels = torch.tensor(np.vectorize(self.subclass_mapper.get)(targets_b.cpu().numpy())).type(torch.LongTensor).to(self.device)
                            # gate_loss = self.criterion(gate_outputs, gate_labels)
                            gate_loss = self.mixup_criterion(self.criterion, gate_outputs, gate_labels, mix_labels, lam)
                            # gate_loss += self.criterion(gate_outputs_ori, ori_labels)

                            running_gate_loss.append(gate_loss.item())
                            loss += gate_loss

                            gate_preds = torch.argmax(gate_outputs.data, 1)
                            # gate_preds_ori = torch.argmax(gate_outputs_ori.data, 1)
                            gate_correct += (lam * gate_preds.eq(gate_labels).cpu().sum().float() + (1 - lam) * gate_preds.eq(mix_labels).cpu().sum().float())
                            # gate_correct += gate_preds_ori.eq(ori_labels).cpu().sum().float()

                        loss.backward()
                        gate_optimiser.step()
                        gate_optimiser.zero_grad()

                        predicted = torch.argmax(outputs.data, 1) if task == 0 else torch.argmax(pred_all_classes, 1)
                        pred_correct += (lam * predicted.eq(targets_a).cpu().sum().float() + (1 - lam) * predicted.eq(targets_b).cpu().sum().float())

                        # predicted = torch.argmax(outputs_ori.data, 1) if task == 0 else torch.argmax(pred_all_classes_ori, 1)
                        # pred_correct += predicted.eq(y).cpu().sum().float()

                        # ypreds.extend(predicted.detach().cpu().tolist())
                        # ytrue.extend(y_mix.detach().cpu().tolist())
                        
                        dataset_len += x.size(0) #+ inputs.size(0)
                
                    # early_stop(loss, self.model)

                    train_loss.append(np.average(running_train_loss))
                    gate_loss_.append(np.average(running_gate_loss))
                    if (epoch + 1) % 10 == 0:                                
                        print(f"STEP-2\tEpoch: {epoch+1}/{self.epochs}\tclassification_loss: {train_loss[-1]:.4f}\tgate_loss: {gate_loss_[-1]:.4f}\tstep2_classification_accuracy: {(100 * pred_correct.item() / dataset_len):.4f}\tstep_2_gate_accuracy: {100 * (gate_correct / dataset_len):.4f}")
                    # if early_stop.early_stop:
                    #     print(f"Early stopping. Exit epoch {epoch+1}")
                    #     break                    
                print("FINISH STEP 2")                
            self.model.unfreeze_all()

            # print("=====GATE WEIGHT END=====")
            # self.model.calculate_gate_norm()
            # print("========================")        

            if val_loaders:
                self.model.eval()
                val_loss_ = []                
                for nt, valloader in enumerate(val_loaders):
                    
                    running_val_loss = []
                    dataset_len = 0
                    gate_correct = 0
                    ypreds, ytrue = [], []
                    for i, (x, y) in enumerate(valloader):                        
                        x = x.to(self.device)
                        y = y.to(self.device)

                        with torch.no_grad():
                            outputs, gate_outputs = self.model(x, task=nt)

                            if task > 0:
                                bias_outputs = []
                                prev = 0
                                for i, num_class in enumerate(self.previous_task_nums):
                                    outs = outputs[:, prev:(prev+num_class)]
                                    bias_outputs.append(self.model.bias_forward(task, outs))
                                    prev += num_class
                                old_cls_outputs = torch.cat(bias_outputs, dim=1)
                                new_cls_outputs = self.model.bias_forward(task, outputs[:, prev:])
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
                                

                            running_val_loss.append(loss.item())
                            dataset_len += x.size(0)

                        predicted = torch.argmax(outputs.data, 1)
                        ypreds.extend(predicted.detach().cpu().tolist())
                        ytrue.extend(y.detach().cpu().tolist())

                    val_loss_.append(np.average(running_val_loss))
                    task_accuracy = 100 * accuracy_score(ytrue, ypreds)
                    print(f"\tTask-{nt}\tval_loss: {val_loss_[-1]:.4f}\tval_accuracy: {task_accuracy:.4f}\tgate_accuracy: {100 * (gate_correct / dataset_len):.4f}")

                    if self.metric:
                        self.metric.add_accuracy(task, task_accuracy)
                self.val_loss.append(val_loss_)

            if self.metric:
                self.metric.add_forgetting(task)
            print()

            self.seen_cls += new_cls
            self.previous_task_nums.append(self.dataset[task]["ncla"])

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

    def update_buffer(self, task, uniform=True):
        current_dataset = self.dataset[task]
        self.replay.update_buffer(current_dataset, uniform=True)

    def get_uniform_mixup(self, task):
        x_train, y_train = self.replay.buffer["x"], self.replay.buffer["y"]
        x_train, y_train = torch.stack(x_train), torch.tensor(y_train)

        mix_inputs, targets_true, targets_new, lam = self.mixup_data(x_train, y_train)
        ori_dataloader = self._get_dataloader(x_train, targets_true.numpy(), shuffle=False, batch_size=64)   # shuffle=False so they both can be compared
        mix_dataloader = self._get_dataloader(mix_inputs, targets_new.numpy(), shuffle=False, batch_size=64)

        # print("\tORI_CLASS_COUNTER:", sorted(Counter(y_train.numpy()).items()))
        # print("\tMIX_CLASS_COUNTER:", sorted(Counter(targets_new.numpy()).items()))

        return ori_dataloader, mix_dataloader, lam

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
    
    def mixup_data(self, x, y, alpha=1.0, use_cuda=True):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam


    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def _get_dataloader(self, x, y, shuffle=False, batch_size=None):
        _dataset = BaseDataset(x, y, self.transform, self.cls2idx)
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