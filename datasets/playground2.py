import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import moe

# from layer import MoE

from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, TensorDataset, DataLoader

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc3 = nn.Linear(768, 2)
#         self.fc3 = MoE(
#             hidden_size=2,
#             expert=self.fc3,
#             num_experts=5,
#             k=2,
#             min_capacity=0,
#             noisy_gate_policy=None)
#         # self.fc4 = nn.Linear(768, 10)

#     def forward(self, x):
#         x, gate_loss, _ = self.fc3(x)
#         # x = self.fc4(x)
#         return x, gate_loss

# def test_sharded_moe():
#     net = Net()
#     inputs = torch.randn(3, 768)
#     outputs, gate_loss = net(inputs)
#     print(outputs)
#     print(gate_loss)

def test_moe():
    # train_embedding_path = "cifar10_train_embedding.pt"
    # test_embedding_path = "cifar10_test_embedding.pt"

    train_embedding_path = "cifar100_coarse_train_embedding.pt"
    test_embedding_path = "cifar100_coarse_test_embedding.pt"

    print("loading train and testset")
    trainset = torch.load(train_embedding_path)
    testset = torch.load(test_embedding_path)

    train_tensor, train_label = ungroup_dataset(trainset)
    test_tensor, test_label = ungroup_dataset(testset)

    train_dataset = TensorDataset(train_tensor, train_label)
    test_dataset = TensorDataset(test_tensor, test_label)

    train_loader = DataLoader(train_dataset, batch_size=64,
                              shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64,
                             shuffle=False)

    kind = "map"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # net = moe.MoE(input_size=768, expert_output_size=2, output_size=10, num_experts=5, hidden_size=256, noisy_gating=True, k=1, kind=kind)
    net = moe.MoE(input_size=768, expert_output_size=4, output_size=20, num_experts=5, hidden_size=256, noisy_gating=True, k=1, kind=kind)
    net = net.to(device)

    epochs = 10
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    ypreds, ytrue = [], []
    net.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimiser.zero_grad()
            outputs, gate_loss, expert_losses, _ = net(inputs, labels)
            loss = criterion(outputs, labels)
            total_loss = loss + gate_loss + sum(expert_losses)
            total_loss.backward()
            optimiser.step()

            running_loss += loss.item()

            predicted = torch.argmax(outputs.data, 1)
            ypreds.extend(predicted.cpu().numpy().tolist())
            ytrue.extend(labels.cpu().numpy().tolist())

            if (i+1) % 100 == 0:
                print(f"Epoch: {epoch+1}/{epochs}\tloss: {np.average(running_loss):.2f}\taccuracy: {(100 * accuracy_score(ytrue, ypreds)):.2f}")
        print()
    print("FINISH TRAINING")

    ypreds, ytrue = [], []
    net.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs, _, _, gates = net(inputs, labels)
            # if kind == "original":
            #     _, predicted = torch.max(outputs.data, 1)
            # else:
            #     predicted = torch.argmax(outputs.data, 1)
            # _, predicted = torch.max(outputs.data, 1)
            predicted = torch.argmax(outputs.data, 1)
            p_temp = predicted.cpu().numpy().tolist()
            r_temp = labels.cpu().numpy().tolist()
            if i == len(test_loader)-1:
                for i, gate in enumerate(gates):
                    print(f"real: {r_temp[i]}\npred: {p_temp[i]}\ngates: {gate.detach().cpu().numpy()}\noutput: {outputs[i].detach().cpu().numpy()}\n")
                print()
            # exit()
            # print(f"predicted: {predicted.cpu().numpy().tolist()}")
            # print(f"real: {labels.cpu().numpy().tolist()}")
            ypreds.extend(predicted.cpu().numpy().tolist())
            ytrue.extend(labels.cpu().numpy().tolist())

    print(f"Accuracy on test set: {(100 * accuracy_score(ytrue, ypreds)):.2f}")

def ungroup_dataset(dataset):
    """
    Given a dataset in the form of:
        {"data": [
            tensor,
            label
        ],
        "classes": []}

    return [tensor], [label]
    """
    tensors, labels = [], []
    for tensor, label in dataset["data"]:
        tensors.append(tensor)
        labels.append(torch.tensor(label))
    return torch.stack(tensors), torch.stack(labels)

if __name__ == "__main__":
    test_moe()