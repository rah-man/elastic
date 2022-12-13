import math
import torch
import torch.nn as nn

from model import SingleMLP
from torch.autograd import Variable

class Strategy:
    def __init__(self):
        self.model = None

    def calculate_loss(self):
        pass

class LwF(Strategy):
    def __init__(self, lambda_=1.6, temperature=2):
        super().__init__()
        self.lambda_ = lambda_
        self.temperature = temperature

    def calculate_loss(self, current_dist, prev_dist, seen_cls):
        # take the current model outputs for old classes
        logits_dist = current_dist[:, :seen_cls]
        log_p = torch.log_softmax(logits_dist / self.temperature, dim=1)
        q = torch.softmax(prev_dist / self.temperature, dim=1)
        distil_loss = nn.functional.kl_div(log_p, q, reduction="batchmean")
        return distil_loss
