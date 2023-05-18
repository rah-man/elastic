from torch import nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    """Simple MLP with one hidden layer."""

    def __init__(self, in_features=768, num_classes=10, **kwargs):
        super().__init__()
        # main part of the network
        self.fc1 = nn.Linear(in_features=in_features, out_features=512)
        # last classifier layer (head) with as many outputs as classes
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)

        # and `head_var` with the name of the head, so it can be removed when doing incremental learning experiments
        self.head_var = 'fc2'

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out


def simpleMLP(pretrained=False, **kwargs):
    if pretrained:
        raise NotImplementedError
    return SimpleMLP(**kwargs)
