import torch
import torch.nn as nn

class SingleMLP(nn.Module):
    def __init__(self, in_features=768, out_features=10, bias=True):
        """
        in_features: the number of input features = 768 following ViT
        out_features: the number of classes (10 for CIFAR10)
        """
        super().__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

    def forward(self, x):
        return self.fc(x)

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