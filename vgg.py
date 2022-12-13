import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.models as models

from moe import MoE
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                transforms.Grayscale(num_output_channels=3),
                                transforms.ToTensor(), 
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

class VGGMoE(nn.Module):
    def __init__(self, output_size, num_experts, hidden_size, k=4):
        super().__init__()
        vgg = models.vgg16()
        self.vgg_features = vgg.features
        self.vgg_avgpool = vgg.avgpool
        self.flatten = nn.Flatten()
        self.moe = MoE(input_size=25088, output_size=output_size, num_experts=num_experts, hidden_size=hidden_size, k=k)

    def forward(self, x):
        x = self.vgg_features(x)
        x = self.vgg_avgpool(x)
        x = self.flatten(x)
        output, moe_loss = self.moe(x)
        return output, moe_loss


def train(trainloader, model, loss_fn, optim, device, print_step, weights=[0.9, 0.001]):
    steps, running_loss, moe_loss_, epoch_loss = 0, [], [], []
    for images, labels in iter(trainloader):
        images = images.to(device)
        labels = labels.to(device)

        optim.zero_grad()

        outputs, moe_loss = model(images)
        loss = loss_fn(outputs, labels)
        total_loss = weights[0] * loss + weights[1] * moe_loss # 0.9 * prediction_loss + 0.001 * gate_loss (smaller fraction as it converges quickly)
        total_loss.backward()
        optim.step()

        running_loss.append(loss.item())
        moe_loss_.append(moe_loss.item())
        steps += 1

        if (steps+1) % print_step == 0:
            print(f"\tSteps: {steps+1}/{len(trainloader)}\n\t\tLoss: {np.average(running_loss):.4f}\n\t\tMoE Loss: {np.average(moe_loss_):.4f}")
            running_loss, moe_loss_ = [], []

def eval(testloader, model, loss_fn, device):
    total_loss, moe_loss_, epoch_loss = [], [], []
    for images, labels in iter(testloader):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs, moe_loss = model(images)
            loss = loss_fn(outputs, labels)

        total_loss.append(loss.item())
        moe_loss_.append(moe_loss.item())

    print(f"\tEvaluation\n\t\tLoss: {np.average(total_loss):.4f}\n\t\tMoE Loss: {np.average(moe_loss_):.4f}")

def train_loop(trainloader, testloader, model, criterion, optimiser, epochs, device, print_step=100, scheduler=None, writer=None, weights=[0.9, 0.001]):
    model.to(device)

    for epoch in range(epochs):
        # train step
        model.train()
        print(f"Epoch {epoch+1}/{epochs}")
        train(trainloader, model, criterion, optimiser, device, print_step, weights)

        # eval step
        model.eval()
        eval(testloader, model, criterion, device)

        # update learning rate ?
        if scheduler:
            scheduler.step()

def predict(model, device, testloader):
    model.to(device)
    model.eval()

    y_preds, y_true = [], []
    
    for images, labels in iter(testloader):
        images = images.to(device)

        with torch.no_grad():
            outputs = model(images)
            if type(outputs) is tuple: # tuple if using MoE, else using original VGG
                outputs = outputs[0]
                outputs = outputs.max(1)[1] # as log_softmax has calculated the probability
            else:
                outputs = torch.argmax(F.softmax(outputs, dim=1), dim=1)
        
        y_preds.extend(outputs.cpu().numpy().tolist())
        y_true.extend(labels.tolist())

    return y_preds, y_true

if __name__ == "__main__":
    with_fc = False
    output_size = 10 # MoE's output, also last classification layer, i.e. with_fc = False
    # output_size = 1000 # MOE's output, not the classification layer output, i.e. with_fc = True
    num_experts = 3 
    hidden_size = 2048 # MoE's hidden unit size
    epochs = 2
    lr = 0.001
    p = 0.3 # dropout probability
    k = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = VGGMoE(output_size, num_experts, hidden_size, k=k).to(device)
    criterion = nn.NLLLoss() # as Rau's MLP outputs are in log_softmax
    optimiser = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = lr_scheduler.ExponentialLR(optimiser, gamma=0.5)

    weights = [.9, .001]

    train_loop(trainloader, testloader, model, criterion, optimiser, epochs, device, print_step=300, scheduler=scheduler, weights=weights)