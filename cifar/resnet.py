import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.models as models

from torchvision import datasets, transforms

mean = [(0.4914, 0.4822, 0.4465), (0.485, 0.456, 0.406)]
std = [(0.2023, 0.1994, 0.2010), (0.229, 0.224, 0.225)]

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean[0], std=std[0]),
])

batch_size = 128
data_path = "../CIFAR_data/"
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
device = "cuda" if torch.cuda.is_available() else "cpu"

trainset = datasets.CIFAR10(
    root=data_path, train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True)

testset = datasets.CIFAR10(
    root=data_path, train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False)


class Resnet(nn.Module):
    def __init__(self, output_size=1000):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, output_size)

    def forward(self, x):        
        return self.resnet(x)

def train_loop(trainloader, testloader, model, criterion, optimiser, epochs, device, print_step=100, scheduler=None, grad_clip=None, writer=None, weights=[0.9, 0.001]):
    model.to(device)

    for epoch in range(epochs):
        # train step
        model.train()
        print(f"Epoch {epoch+1}/{epochs}")
        train(trainloader, model, criterion, optimiser, device, print_step, weights, scheduler=scheduler, grad_clip=grad_clip)

        # eval step
        model.eval()
        eval(testloader, model, criterion, device)

        # update learning rate ?
        if scheduler:
            scheduler.step()

def train(trainloader, model, loss_fn, optim, device, print_step, weights=[0.9, 0.001], scheduler=None, grad_clip=None):
    steps, total_loss = 0, []
    for images, labels in iter(trainloader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        
        if grad_clip:
            nn.utils.clip_grad_value_(model.parameters(), grad_clip)

        optim.step()
        optim.zero_grad()

        if scheduler:
            scheduler.step()

        total_loss.append(loss.item())
        steps += 1

        if (steps+1) % print_step == 0:
            print(f"\tSteps: {steps+1}/{len(trainloader)}\tLoss: {np.average(total_loss):.4f}")

def eval(testloader, model, loss_fn, device):
    total_loss = []
    for images, labels in iter(testloader):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(images)
            loss = loss_fn(outputs, labels)

        total_loss.append(loss.item())

    print(f"\tEvaluation\tLoss: {np.average(total_loss):.4f}")

def predict(model, device, testloader):
    model.to(device)
    model.eval()

    y_preds, y_true = [], []
    
    for images, labels in iter(testloader):
        images = images.to(device)

        with torch.no_grad():
            outputs = model(images)
            outputs = torch.argmax(outputs, dim=1)
        
        y_preds.extend(outputs.cpu().numpy().tolist())
        y_true.extend(labels.tolist())

    return y_preds, y_true

if __name__ == "__main__":
    output_size = 10 # CIFAR10
    epochs = 20
    max_lr = 0.1
    weight_decay = 1e-4
    grad_clip = 0.1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Resnet(output_size=output_size)    
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.OneCycleLR(optimiser, max_lr, epochs=epochs, steps_per_epoch=len(trainloader))
    # optimiser = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)

    train_loop(trainloader, testloader, model, criterion, optimiser, epochs, device, print_step=300, scheduler=scheduler, grad_clip=grad_clip)