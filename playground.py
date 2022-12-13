import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import moe

from base import get_data
# from layer import MoE

from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import Dataset, TensorDataset, DataLoader


def test_moe():
    train_embedding_path = "cifar10_train_embedding.pt"
    test_embedding_path = "cifar10_test_embedding.pt"

    dataset, class_order = get_data(
        train_embedding_path, 
        test_embedding_path, 
        num_tasks=5, 
        validation=0.2,
        seed=42)


    cls2idx = {v: k for k, v in enumerate(class_order)}
    idx2cls = {k: v for k, v in enumerate(class_order)}
    print(class_order)
    print(cls2idx)
    print(idx2cls)

    for i in range(len(dataset)):
        print(dataset[i]["name"])
        # print(dataset[i]["train"]["x"])
        # print(dataset[i]["val"]["y"])
        # print(dataset[i]["test"]["x"][0])
        print(dataset[i]["classes"])
        print()

    exit()

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = moe.MoE(input_size=768, output_size=10, num_experts=10, hidden_size=256, noisy_gating=True, k=4)
    net = net.to(device)

    epochs = 5
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    net.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimiser.zero_grad()
            outputs, aux_loss = net(inputs)
            loss = criterion(outputs, labels)
            total_loss = loss + aux_loss
            total_loss.backward()
            optimiser.step()

            running_loss += loss.item()
            if (i+1) % 100 == 0:
                print(f"Epoch: {epoch+1}/{epochs}, loss: {np.average(running_loss):.4f}")
        print()
    print("FINISH TRAINING")

    ypreds, ytrue = [], []
    net.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs, _ = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            ypreds.extend(predicted.cpu().numpy().tolist())
            ytrue.extend(labels.cpu().numpy().tolist())

    print(f"Accuracy: {(100 * accuracy_score(ytrue, ypreds)):.4f}")
    print("Confusion Matrix")
    print(confusion_matrix(ytrue, ypreds))

def generate_weight_noise(inp_size=768, n_expert=5):
    torch.manual_seed(42)
    w_gate = torch.rand(inp_size, n_expert)
    torch.manual_seed(43)
    w_noise = torch.rand(inp_size, n_expert)
    return w_gate, w_noise

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