import math
import torch
import torch.nn as nn

from model import SingleMLP
from torch.autograd import Variable

class Regulariser:
    def __init__(self):
        self.model = None

    def update_model(self):
        pass

    def calculate_loss(self):
        pass

class LwF(Regulariser):
    def __init__(self, lambda_=1.6, temperature=2, device="cpu"):
        super().__init__()
        self.lambda_ = lambda_
        self.temperature = temperature
        self.device = device

    def update_model(self, seen_cls, new_cls):
        self.seen_cls = seen_cls
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

    def calculate_loss(self, current_dist, prev_dist):
        # take the current model outputs for old classes
        logits_dist = current_dist[:, :self.seen_cls]
        log_p = torch.log_softmax(logits_dist / self.temperature, dim=1)
        q = torch.softmax(prev_dist / self.temperature, dim=1)
        distil_loss = nn.functional.kl_div(log_p, q, reduction="batchmean")
        return distil_loss

    def _multiclass_crossentropy(self, logits_dist, prev_dist, temperature, device):
        labels = Variable(prev_dist, requires_grad=False).to(device)
        outputs = torch.log_softmax(logits_dist / temperature, dim=1)
        labels = torch.softmax(labels / temperature, dim=1)
        outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
        outputs = -torch.mean(outputs, dim=0, keepdim=False)
        return Variable(outputs, requires_grad=True).to(device)

    def _kaiming_normal_init(self, model):
        if isinstance(model, nn.Conv2d):
            nn.init.kaiming_normal_(model.weight, nonlinearity="relu")
        if isinstance(model, nn.Linear):
            nn.init.kaiming_normal_(model.weight, nonlinearity="sigmoid")
