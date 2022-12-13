import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.models as models

from torchvision import datasets, transforms

import model, utils

if __name__ == "__main__":
    output_size = 10 # CIFAR10
    epochs = 160
    max_lr = 0.01
    weight_decay = 1e-4
    grad_clip = 0.1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    resnet18 = model.ResNet(3, model.ResBlock, [2, 2, 2, 2], useBottleneck=False, outputs=output_size)
    criterion = nn.CrossEntropyLoss()
    # optimiser = optim.SGD(resnet18.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    optimiser = optim.Adam(resnet18.parameters(), lr=max_lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.OneCycleLR(optimiser, max_lr, epochs=epochs, steps_per_epoch=len(utils.trainloader))

    utils.train_loop(utils.trainloader, utils.testloader, resnet18, criterion, optimiser, epochs, device, print_step=10, scheduler=scheduler, grad_clip=grad_clip)    

