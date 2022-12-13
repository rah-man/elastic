import torch
import torch.nn as nn

splus = nn.Softplus()
input = torch.randn(2)
output = splus(input)
print(output)

input_size = 1000
num_classes = 20
num_experts = 10
hidden_size = 64
batch_size = 5
k = 4

def dummy_data(batch_size, input_size, num_classes):
    x = torch.rand(batch_size, input_size)
    y = torch.randint(num_classes, (batch_size, 1)).squeeze(1)
    return x, y

x, y =  dummy_data(batch_size, input_size, num_classes)
print(x)
print(x.size())
print(y)
print(y.size())

w_gate = nn.Parameter(torch.zeros(input_size, num_experts))
print(w_gate)
print(w_gate.size())

gates = x @ w_gate
print(gates)
print(gates.size())