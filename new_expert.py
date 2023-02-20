import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal

class Expert(nn.Module):
    def __init__(self, input_size=768, hidden_size=20, output_size=2, projected_output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.mapper = nn.Linear(in_features=output_size, out_features=projected_output_size, bias=False)
        self.batchnorm = nn.BatchNorm1d(num_features=hidden_size, affine=True)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = F.relu(self.batchnorm(self.fc1(x)))
        out = self.mapper(self.fc2(out))
        return out

class BiasLayer(torch.nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        # Initialize alpha and beta with requires_grad=False and only set to True during Stage 2
        self.alpha = torch.nn.Parameter(torch.ones(1, requires_grad=False))
        self.beta = torch.nn.Parameter(torch.zeros(1, requires_grad=False))

    def forward(self, x):
        return self.alpha * x + self.beta        

class DynamicExpert(nn.Module):
    def __init__(self, input_size=768, hidden_size=20, total_cls=100):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.total_cls = total_cls
        self.gate = None
        self.experts = None
        self.bias_layers = None

    def expand_expert(self, seen_cls, new_cls):
        self.seen_cls = seen_cls
        self.new_cls = new_cls

        if not self.experts:
            gate = nn.Linear(in_features=self.input_size, out_features=1)
            # gate = nn.Sequential(
            #     nn.Linear(in_features=self.input_size, out_features=1000),
            #     nn.ReLU(),
            #     nn.Linear(in_features=1000, out_features=1)
            # )
            # for module in gate:
            #     if isinstance(module, nn.Linear):
            #         torch.nn.init.xavier_uniform_(module.weight)

            experts = nn.ModuleList([Expert(input_size=self.input_size, hidden_size=self.hidden_size, output_size=new_cls, projected_output_size=new_cls)])
            self.bias_layers = nn.ModuleList([BiasLayer()])
            self.num_experts = len(experts)       
        else:
            # UPDATE THE GATE OUTPUT
            # UPDATE THE MAPPER LAYER FOR EACH OF THE PREVIOUS EXPERT
            # 1. old_mapper.out_features = old_mapper.out_features + self.new_cls
            # 2. update zero weights for corresponding experts\

            # old_gate = copy.deepcopy(self.gate)
            # with torch.no_grad():
            #     old_weight_mean = torch.mean(old_gate.weight, 0)
            
            gate = nn.Linear(in_features=self.input_size, out_features=self.num_experts+1)            
            # old_size = old_gate.weight.size(0)
            # with torch.no_grad():
            #     gate.weight[:old_size, :] = old_gate.weight
                # gate.weight[old_size:, :] = old_weight_mean # the new weight is the mean of the old weights

            # gate = nn.Sequential(
            #     nn.Linear(in_features=self.input_size, out_features=1000),
            #     nn.ReLU(),
            #     nn.Linear(in_features=1000, out_features=self.num_experts+1)
            # )
            # for module in gate:
            #     if isinstance(module, nn.Linear):
            #         torch.nn.init.xavier_uniform_(module.weight)
            # with torch.no_grad():
            #     gate[0].weight = old_gate[0].weight
            #     for i in range(self.num_experts):
            #         gate[2].weight[i, :] = old_gate[2].weight[i, :]

            experts = copy.deepcopy(self.experts)
            experts.append(Expert(input_size=self.input_size, hidden_size=self.hidden_size, output_size=new_cls, projected_output_size=new_cls))
            self.num_experts = len(experts)
            for expert_index, module in enumerate(experts):
                weight = module.mapper.weight
                input_size = module.mapper.in_features
                module.mapper = nn.Linear(in_features=input_size, out_features=(seen_cls + new_cls), bias=False)

                with torch.no_grad():
                    all_ = {i for i in range(seen_cls + new_cls)}
                    removed_ = all_ - {i for i in range(new_cls * expert_index, new_cls * expert_index + new_cls)}
                    module.mapper.weight[new_cls * expert_index:new_cls * expert_index + new_cls, :] = weight if weight.size(0) <= new_cls else weight[new_cls * expert_index:new_cls * expert_index + new_cls, :]
                    module.mapper.weight[list(removed_)] = 0.

            self.bias_layers.append(BiasLayer())
        
        # self.k = min(k, self.num_experts) if k > self.num_experts else k
        self.gate = gate
        self.experts = experts

        gate_total_params = sum(p.numel() for p in gate.parameters())
        print(f"task-{self.num_experts-1} GATE_TOTAL_PARAMS: {gate_total_params}")
        expert_total_params = sum(p.numel() for p in experts.parameters())
        print(f"task-{self.num_experts-1} EXPERT_TOTAL_PARAMS: {expert_total_params}")
        bias_total_params = sum(p.numel() for p in self.bias_layers.parameters())
        print(f"task-{self.num_experts-1} bias_TOTAL_PARAMS: {bias_total_params}")                    

    def calculate_gate_norm(self):
        w1 = nn.utils.weight_norm(self.gate, name="weight")
        print(w1.weight_g)
        nn.utils.remove_weight_norm(w1)

    def bias_forward(self, task, output):
        """Modified version from FACIL"""
        return self.bias_layers[task](output)

    def freeze_previous(self):
        for i in range(len(self.experts) - 1):
            e = self.experts[i]
            for param in e.parameters():
                param.requires_grad = False

    def freeze_all(self):
        for e in self.experts:
            for param in e.parameters():
                param.requires_grad = False

    def set_gate(self, grad):
        for name, param in self.named_parameters():
            if name == "gate":
                param.requires_grad = grad

    def unfreeze_all(self):
        for e in self.experts:
            for param in e.parameters():
                param.requires_grad = True

    def forward(self, x, y, task=None, train_step=2):
        # print("\tx.size():", x.size())
        
        gate_outputs = None
        if train_step == 1:
            expert_outputs = self.experts[task](x)
            # print("\tstep1 expert_outputs.size():", expert_outputs.size())
        else:
            gate_outputs = self.gate(x)
            gate_outputs_uns = torch.unsqueeze(gate_outputs, 1)
            
            expert_outputs = [self.experts[i](x) for i in range(self.num_experts)]
            expert_outputs = torch.stack(expert_outputs, 1)
            expert_outputs = gate_outputs_uns@expert_outputs
            expert_outputs = torch.squeeze(expert_outputs)
            # print("\tstep2 expert_outputs.size():", expert_outputs.size())
        

        # print("\texpert_outputs.size():", expert_outputs.size())
        # python new_expert_trainer.py -d 2 -b 256 -s 2 -m 500 -e 200 -n 20 -p "z.pkl"

        return expert_outputs, gate_outputs
        