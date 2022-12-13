import copy
import torch
import torch.nn as nn

from abc import ABC, abstractmethod

class SingleMLP(nn.Module):
    """
    An obsolete class that is not used anymore
    """
    def __init__(self, in_features=768, out_features=10, bias=True):
        """
        in_features: the number of input features = 768 following ViT
        out_features: the number of classes (10 for CIFAR10)
        """
        super().__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

    def forward(self, x):
        return self.fc(x)

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass
    
    @abstractmethod
    def update_model(self, seen_cls, new_cls, cls_labels=None):
        pass
    
    @abstractmethod
    def not_none(self):
        pass

class IncrementalModel(BaseModel):
    def __init__(self, in_features=768):
        super().__init__()
        self.seen_cls = 0
        self.new_cls = 0
        self.in_features = in_features
        self.fc = None

    def forward(self, x):
        return self.fc(x)

    def update_model(self, seen_cls, new_cls, cls_labels=None):
        self.seen_cls = seen_cls
        self.new_cls = new_cls

        if not self.fc:
            fc = nn.Linear(in_features=self.in_features, out_features=new_cls)
        else:
            with torch.no_grad():
                old_weights = self.fc.weight
                old_biases = self.fc.bias
            out_features = seen_cls + new_cls
            
            fc = nn.Linear(in_features=self.in_features, out_features=out_features)

            fc.weight.data[:seen_cls] = old_weights
            fc.bias.data[:seen_cls] = old_biases
            
        self.fc = fc

    def not_none(self):
        return self.fc != None

class MultiHeadModel(BaseModel):
    def __init__(self, in_features=768):
        super().__init__()
        self.in_features = in_features
        self.seen_cls = 0
        self.new_cls = 0
        self.task = 0
        self.classifiers = nn.ModuleDict()
        self.idx2cls = []
        self.cls2idx = []
        self.is_training = True

    def update_model(self, seen_cls, new_cls, cls_labels=None):
        self.seen_cls = seen_cls
        self.new_cls = new_cls

        fc = nn.Linear(in_features=self.in_features, out_features=new_cls)
        taskid = str(self.task)
        self.classifiers[taskid] = fc
        self.task += 1

        self.cls2idx.append({v: k for k, v in enumerate(cls_labels)})
        self.idx2cls.append({k: v for k, v in enumerate(cls_labels)})

        # not sure how to implement this
        # for now, if is_training = True, freeze tasks 0 .. self.task - 1 so their weights don't change
        # btw, no matter what is_training is, always freeze the previous tasks' weights
        # i.e., in the next training phase, the previous heads are not updated anymore
        for taskid in range(self.task-1):
            self.classifiers[str(taskid)].weight.requires_grad = False

    def forward(self, x):
        res = {}
        for taskid in range(self.task):
            res[taskid] = self.classifiers[str(taskid)](x)
        return res

    def not_none(self):
        return len(self.classifiers) != 0

    def check_requires_grad(self):
        for taskid in range(self.task):
            print(f"{str(taskid)}: {self.classifiers[(str(taskid))].weight.requires_grad}")


class BiasLayer(nn.Module):
    def __init__(self, device="cpu"):
        super(BiasLayer, self).__init__()
        self.beta = nn.Parameter(torch.ones(1, requires_grad=True, device=device))
        self.gamma = nn.Parameter(torch.zeros(1, requires_grad=True, device=device))

    def forward(self, x):
        return self.beta * x + self.gamma

    def printParam(self, i):
        print(f"Bias: {i}\tBeta: {self.beta.item()}\tgamma: {self.gamma.item()}")

    def get_beta(self):
        return self.beta

    def get_gamma(self):
        return self.gamma

    def set_beta(self, new_beta):
        self.beta = new_beta

    def set_gamma(self, new_gamma):
        self.gamma = new_gamma

    def set_grad(self, bool_value):
        self.beta.requires_grad = bool_value
        self.gamma.requires_grad = bool_value        

if __name__ == "__main__":
    num_tasks = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bias_layers = [BiasLayer(device=device) for i in range(num_tasks)]
    for i, layer in enumerate(bias_layers):
        layer.printParam(i)

    print("======")    
    cnt_bias_corr_params = 0
    for layer in bias_layers:
        print(sum(p.numel() for p in layer.parameters()))
        cnt_bias_corr_params += sum(p.numel() for p in layer.parameters())        
    print(cnt_bias_corr_params)