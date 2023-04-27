import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kneebow.rotor import Rotor
from self_attention_cv import MultiHeadSelfAttention, SelfAttention
from torch.distributions.normal import Normal

class Expert(nn.Module):
    def __init__(self, input_size=768, hidden_size=20, output_size=2, projected_output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.mapper = nn.Linear(in_features=output_size, out_features=projected_output_size, bias=False)
        # self.batchnorm = nn.BatchNorm1d(num_features=hidden_size, affine=True)
        self.instancenorm = nn.InstanceNorm1d(num_features=hidden_size)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        # out = F.relu(self.batchnorm(self.fc1(x)))
        out = F.relu(self.instancenorm(self.fc1(x)))
        out = self.mapper(self.fc2(out))
        return out

    def off_map(self):
        for param in self.mapper.parameters():
            param.requires_grad = False

class BiasLayer(torch.nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        # Initialize alpha and beta with requires_grad=False and only set to True during Stage 2
        self.alpha = torch.nn.Parameter(torch.ones(1, requires_grad=False))
        self.beta = torch.nn.Parameter(torch.zeros(1, requires_grad=False))

    def forward(self, x):
        return self.alpha * x + self.beta        

class GateA(nn.Module):
    def __init__(self, input_size=768, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=input_size//2)
        # self.norm = nn.InstanceNorm1d(num_features=input_size//2)
        # self.relu = nn.ReLU()
        self.attention = SelfAttention(dim=input_size//2)
        # self.attention = MultiHeadSelfAttention(dim=input_size//2)
        self.fc2 = nn.Linear(in_features=input_size//2, out_features=output_size)

    def forward(self, x):
        out = self.fc1(x)
        # out = self.norm(out)
        # out = self.relu(out)
        out = torch.unsqueeze(out, 1)
        out = self.attention(out)
        out = torch.squeeze(out)
        out = self.fc2(out)        
        return out

class GateB(nn.Module):
    def __init__(self, input_size=768, output_size=1):
        super().__init__()
        # self.attention = SelfAttention(dim=input_size)
        self.att1 = SelfAttention(dim=input_size)
        self.fc1 = nn.Linear(in_features=input_size, out_features=input_size//2)
        self.att2 = SelfAttention(dim=input_size//2)
        self.fc2 = nn.Linear(in_features=input_size//2, out_features=output_size)

    def forward(self, x):
        out = torch.unsqueeze(x, 1)
        out = self.att1(out)
        out = torch.squeeze(out)
        out = self.fc1(out)
        out = torch.unsqueeze(out, 1)
        out = self.att2(out)
        out = torch.squeeze(out)
        out = self.fc2(out)        
        return out

class DynamicExpert(nn.Module):
    def __init__(self, input_size=768, hidden_size=20, total_cls=100, class_per_task=20, device="cpu", use_bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.total_cls = total_cls
        self.class_per_task = class_per_task
        self.device = device
        self.gate = None
        self.experts = None
        self.bias_layers = None
        self.all_classes = []
        self.expert_classes = []
        self.cossim = {}
        self.old_weight = {} # for weight distance calculation
        self.old_gate = None
        self.T = 2
        self.use_bias = use_bias

    def expand_gmm(self, this_task_classes):
        if not self.experts:
            # gate = nn.Linear(in_features=self.input_size, out_features=1)
            gate = nn.Sequential(nn.Linear(in_features=self.input_size, out_features=self.input_size//2),
                                # nn.BatchNorm1d(num_features=self.input_size//2, affine=True),
                                nn.InstanceNorm1d(num_features=self.input_size//2),
                                nn.ReLU(),
                                # SelfAttention(dim=self.input_size//2),
                                nn.Linear(in_features=self.input_size//2, out_features=1))
            # gate = GateA(input_size=self.input_size, output_size=1)
            # gate = GateB(input_size=self.input_size, output_size=1)
            hidden_size = int((self.hidden_size / self.class_per_task) * len(this_task_classes))
            experts = nn.ModuleList([Expert(input_size=self.input_size, hidden_size=hidden_size, output_size=len(this_task_classes), projected_output_size=len(this_task_classes))])

            if self.use_bias:
                self.bias_layers = nn.ModuleList([BiasLayer()])

            self.num_experts = len(experts)
            self.all_classes.extend(sorted(this_task_classes))
            self.expert_classes.append(sorted(this_task_classes))
        else:
            self.old_gate = copy.deepcopy(self.gate)
            self.old_gate.eval()

            # gate = nn.Linear(in_features=self.input_size, out_features=self.num_experts+1)
            gate = nn.Sequential(nn.Linear(in_features=self.input_size, out_features=self.input_size//2),
                                # nn.BatchNorm1d(num_features=self.input_size//2, affine=True),
                                nn.InstanceNorm1d(num_features=self.input_size//2),
                                nn.ReLU(),
                                # SelfAttention(dim=self.input_size//2),
                                nn.Linear(in_features=self.input_size//2, out_features=self.num_experts+1))
            # gate = GateA(input_size=self.input_size, output_size=self.num_experts+1)
            # gate = GateB(input_size=self.input_size, output_size=self.num_experts+1)
            
            # copy fc1, attention, and fc2 from old gate
            # with torch.no_grad():
            #     gate.fc1.weight.data = self.old_gate.fc1.weight
            #     gate.attention.to_qvk.data = self.old_gate.attention.to_qvk.weight
            #     # gate.attention.W_0.data = self.old_gate.attention.W_0.weight
            #     # gate.att1.to_qvk.data = self.old_gate.att1.to_qvk.weight
            #     # gate.att1.W_0.data = self.old_gate.att1.W_0.weight
            #     # gate.fc1.weight.data = self.old_gate.fc1.weight
            #     # gate.att2.to_qvk.data = self.old_gate.att2.to_qvk.weight
            #     # gate.att2.W_0.data = self.old_gate.att2.W_0.weight                
            #     old_fc2_size = self.old_gate.fc2.weight.size(0)
            #     gate.fc2.weight.data[:old_fc2_size, :] = self.old_gate.fc2.weight

            hidden_size = int((self.hidden_size / self.class_per_task) * len(this_task_classes))
            experts = copy.deepcopy(self.experts)
            experts.append(Expert(input_size=self.input_size, hidden_size=hidden_size, output_size=len(this_task_classes), projected_output_size=len(this_task_classes)))

            if self.use_bias:
                self.bias_layers.append(BiasLayer())

            self.num_experts = len(experts)
            self.all_classes.extend(sorted(this_task_classes))
            self.expert_classes.append(sorted(this_task_classes))

            low, high = 0, 0
            for expert_index, module in enumerate(experts):
                weight = copy.deepcopy(module.mapper.weight)
                input_size = module.mapper.in_features
                high += len(self.expert_classes[expert_index])

                with torch.no_grad():
                    temp = torch.zeros_like(torch.empty(len(self.all_classes), input_size))
                    if expert_index < len(experts) - 1:
                        # for previous experts, copy the weight following the indices
                        temp[low:high, :] = weight[low:high, :]
                    else:
                        # for new expert, move the random non-zero to the end
                        temp[low:high, :] = weight[:, :]
                    low = high
                    module.mapper = nn.Linear(in_features=input_size, out_features=len(self.all_classes), bias=False)
                    module.mapper.weight.data = temp.data
        
        self.gate = gate
        self.experts = experts

        for i in range(len(self.experts)-1):
            self.experts[i].off_map()

        gate_total_params = sum(p.numel() for p in gate.parameters())
        # print(f"task-{self.num_experts-1} GATE_TOTAL_PARAMS: {gate_total_params}")
        expert_total_params = sum(p.numel() for p in experts.parameters())
        # print(f"task-{self.num_experts-1} EXPERT_TOTAL_PARAMS: {expert_total_params}")
        if self.use_bias:
            bias_total_params = sum(p.numel() for p in self.bias_layers.parameters())
        # print(f"task-{self.num_experts-1} bias_TOTAL_PARAMS: {bias_total_params}")

    def kd_loss(self, x, current_output):
        """Calculate knowledge distillation loss on gate"""
        old_output = self.old_gate(x)   # old_output is one unit behind current output
        t = old_output.size(1)
        # print(f"CURRENT_OUTPUT[:t].size(): {current_output[:t].size()}")
        # print(f"OLD_OUTPUT[:t].size(): {old_output[:t].size()}")
        loss = self.cross_entropy(current_output[:, :t], old_output[:, :t], exp=1.0 / self.T)        
        return loss

    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
        """Calculate cross entropy with temperature scaling"""
        # print(f"outputs.size(): {outputs.size()}")
        # print(f"targets.size(): {targets.size()}")
        out = torch.nn.functional.softmax(outputs, dim=1)
        tar = torch.nn.functional.softmax(targets, dim=1)
        if exp != 1:
            out = out.pow(exp)
            out = out / out.sum(1).view(-1, 1).expand_as(out)
            tar = tar.pow(exp)
            tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
        out = out + eps / out.size(1)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        ce = -(tar * out.log()).sum(1)
        if size_average:
            ce = ce.mean()
        return ce


    def calculate_expert_weight_distance(self):
        # call after subtask
        # take only the index where the value is (like when expanding the mapper) --> NO NEED, AS THIS IS THE FC WEIGHT, NOT THE MAPPER WEIGHT

        cos = nn.CosineSimilarity(dim=1)

        # self.old_weight contains {expert_index: [fc1_weight, fc2_weight]}
        # self.cossim contains {expert_index: [fc1_dist, fc2_dist]} for every expert

        for expert_index, module in enumerate(self.experts):
            temp = self.old_weight.get(expert_index, [])
            if temp:
                # previous weight exists
                fc1_old, fc2_old = temp[0], temp[1]
            else:
                # previous one doesn't exist
                fc1_old, fc2_old = module.fc1.weight.data, module.fc2.weight.data

            fc1_weight, fc2_weight = module.fc1.weight.data, module.fc2.weight.data
            fc1_dist = cos(fc1_old, fc1_weight).sum() / fc1_weight.size(0)
            fc2_dist = cos(fc2_old, fc2_weight).sum() / fc2_weight.size(0)
            fc1_dist = fc1_dist.item()
            fc2_dist = fc2_dist.item()

            self.cossim[expert_index] = [fc1_dist, fc2_dist]
            self.old_weight[expert_index] = [fc1_weight, fc2_weight]

    def print_weight_distance(self):
        for expert_index in sorted(self.cossim.keys()):
            print(f"\tExpert-{expert_index} weight distance: {self.cossim[expert_index]}")
        print()

    def calculate_gate_norm(self):
        w1 = nn.utils.weight_norm(self.gate, name="weight")
        # print(w1.weight_g)
        nn.utils.remove_weight_norm(w1)

    def bias_forward(self, task, output):
        """Modified version from FACIL"""
        return self.bias_layers[task](output)

    def freeze_previous_experts(self):
        for i in range(len(self.experts) - 1):
            e = self.experts[i]
            for param in e.parameters():
                param.requires_grad = False

    def freeze_all_experts(self):
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

    def forward(self, x, task=None, train_step=2):
        gate_outputs = None
        original_expert_outputs = None

        if train_step == 1:
            expert_outputs = self.experts[task](x)
            # original_expert_outputs = copy.deepcopy(expert_outputs) # no need to do this but to make it uniform
            original_expert_outputs = expert_outputs.detach()
        else:
            gate_outputs = self.gate(x)
            gate_outputs_uns = torch.unsqueeze(gate_outputs, 1)
            
            expert_outputs = [self.experts[i](x) for i in range(self.num_experts)]
            # original_expert_outputs = copy.deepcopy(expert_outputs)
            original_expert_outputs = [e.detach() for e in expert_outputs]

            expert_outputs = torch.stack(expert_outputs, 1)
            expert_outputs = gate_outputs_uns@expert_outputs
            expert_outputs = torch.squeeze(expert_outputs)

        return expert_outputs, gate_outputs, original_expert_outputs

    def predict(self, x, type="logits"):
        # UPDATE ON GATE_OUTPUTS REFERENCE WILL ALSO UPDATE THE ORIGINAL VALUES
        self.eval()
        gate_outputs = self.gate(x).cpu()
        original_gate_outputs = torch.clone(gate_outputs).detach()

        if type == "logits":
            # get top-k using elbow based on logits and no scale
            for out in gate_outputs:
                data = out.numpy()
                data_rev = np.array([[i, sorted(data, reverse=True)[i]] for i in range(data.shape[0])])
                rotor = Rotor()
                rotor.fit_rotate(data_rev, scale=False)
                val = data_rev[rotor.get_elbow_index()][-1]
                data[data < val] = 0
        elif type == "softmax":
            # using softmax on logits and no scale
            smax = torch.softmax(gate_outputs, 1)
            for sm, ga in zip(smax, gate_outputs):
                data = sm.numpy()
                data_ = ga
                data_sorted = np.array([[i, sorted(data, reverse=True)[i]] for i in range(data.shape[0])])
                rotor = Rotor()
                rotor.fit_rotate(data_sorted, scale=False)
                data_[data_ < data_[np.where(data == data_sorted[rotor.get_elbow_index()][-1])[0][0]]] = 0        
        else:
            # type could be "one"
            k_idx = torch.argmax(gate_outputs, 1)
            for idx, out in zip(k_idx, gate_outputs):
                out[out != out[idx.item()]] = 0.0

        gate_outputs = gate_outputs.to(self.device)
        gate_outputs_uns = torch.unsqueeze(gate_outputs, 1)

        expert_outputs = [self.experts[i](x) for i in range(self.num_experts)]
        original_expert_outputs = [torch.clone(e).detach() for e in expert_outputs]
        
        expert_outputs = torch.stack(expert_outputs, 1)
        expert_outputs = gate_outputs_uns@expert_outputs
        expert_outputs = torch.squeeze(expert_outputs)

        return expert_outputs, gate_outputs, original_expert_outputs, original_gate_outputs