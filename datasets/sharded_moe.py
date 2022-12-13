import copy
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from time import perf_counter
from torch import Tensor
from typing import Callable, Dict, TYPE_CHECKING, Any, Optional, Tuple, Union, cast

from base import BaseDataset, Extractor, get_data
from model import IncrementalModel, MultiHeadModel
from strategy import LwF, Strategy
from replay import RandomReplay


uniform_map: Dict[torch.device, Callable] = {}
gumbel_map: Dict[torch.device, Callable] = {}
exp_selection_uniform_map: Dict[torch.device, Callable] = {}

class Experts(torch.nn.Module):
    def __init__(self, expert, num_local_experts=1):
        super().__init__()

        self.experts = torch.nn.ModuleList(
            [copy.deepcopy(expert) for i in range(num_local_experts)])
        self.num_local_experts = num_local_experts

        for expert in self.experts:
            for name, param in expert.named_parameters():
                param.allreduce = False

    def forward(self, inputs):
        # chunks = inputs.chunk(self.num_local_experts, dim=1)
        expert_outputs = []

        # for chunk, expert in zip(chunks, self.experts):
        for expert in self.experts:
            out = expert(inputs)
            if type(out) is tuple:
                out = out[0]
            expert_outputs += [out]
        expert_output = torch.cat(expert_outputs, dim=1)

        return expert_output

TUTEL_INSTALLED = False

def multiplicative_jitter(x, device: torch.device, epsilon=1e-2):
    """
    Modified from switch transformer paper. mesh transformers
    Multiply values by a random number between 1-epsilon and 1+epsilon.
    Makes models more resilient to rounding errors introduced by bfloat16.
    This seems particularly important for logits.
    Args:
        x: a torch.tensor
        device: torch.device
        epsilon: a floating point value
    Returns:
        a jittered x.
    """
    if epsilon == 0:
        return x
    uniform = uniform_map.get(device)
    if uniform is None:
        uniform = torch.distributions.uniform.Uniform(
            low=torch.tensor(1.0 - epsilon,
                             device=device),
            high=torch.tensor(1.0 + epsilon,
                              device=device)).rsample  # type: ignore
        uniform_map[device] = uniform
    return x * uniform(x.shape)

def gumbel_rsample(shape: Tuple, device: torch.device) -> Tensor:
    gumbel = gumbel_map.get(device)
    if gumbel is None:
        one = torch.tensor(1.0, device=device)
        zero = torch.tensor(0.0, device=device)
        gumbel = torch.distributions.gumbel.Gumbel(zero, one).rsample  # type: ignore
        gumbel_map[device] = gumbel
    return gumbel(shape)

# einsum rewrites are on par or more performant
# switch can be bubbled up in future
USE_EINSUM = True

# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# See https://arxiv.org/pdf/2006.16668.pdf for details.
def einsum(rule, a, b):
    if USE_EINSUM:
        return torch.einsum(rule, a, b)
    elif rule == 's,se->se':
        return a.reshape(a.shape[0], -1) * b
    elif rule == 'se,sc->sec':
        return a.unsqueeze(2) * b.unsqueeze(1)
    elif rule == 'se,se->s':
        return torch.bmm(a.unsqueeze(1), b.unsqueeze(2)).reshape(-1)
    elif rule == 'sec,sm->ecm':
        s = a.shape[0]
        e = a.shape[1]
        c = a.shape[2]
        m = b.shape[1]
        return torch.matmul(a.reshape(s, -1).t(), b).reshape(e, c, m)
    elif rule == 'sec,ecm->sm':
        return torch.matmul(a.reshape(a.shape[0], -1), b.reshape(-1, b.shape[-1]))
    elif rule == 'ks,ksm->sm':
        k = b.shape[0]
        s = b.shape[1]
        m = b.shape[2]
        # [k, s] -> [s, k] -> [s, 1, k]
        a = a.t().unsqueeze(1)
        # [k,s,m] -> [k, sm] -> [sm, k] -> [s, m, k]
        b = b.reshape(k, -1).t().reshape(s, m, k)
        # bmm([s, 1, k], [s, m, k]^t) -> [s, m, 1]
        return torch.bmm(a, b.transpose(1, 2)).squeeze(2)
    else:
        return torch.einsum(rule, a, b)
        
# The following functions are extracted and scripted
# because otherwise during a torch.jit.trace, the non-Tensor
# values used in the calculations get recorded as constants.
# torch.jit.script coerces them into Tensors and preserves
# their dynamic shapes. This enables ONNX export.
# We can't script the entire top1gating function because it
# includes stateful caching logic which is incompatible with ONNX.

@torch.jit.script
def _capacity(gates: Tensor, capacity_factor: Tensor, min_capacity: Tensor) -> Tensor:
    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    # to(torch.int64) works around a bug in torch.onnx.export:
    # it should cast k to int64 when converting torch.topk but it doesn't.
    capacity = torch.ceil((num_tokens / num_experts) * capacity_factor).to(torch.int64)
    if capacity < min_capacity:
        capacity = min_capacity.to(torch.int64)
    return capacity

@torch.jit.script
def _one_hot_to_float(x, num_classes):
    return F.one_hot(x, num_classes=num_classes).float()

def top2gating(logits: Tensor,
               capacity_factor: float,
               min_capacity: int) -> Tuple[Tensor,
                                           Tensor,
                                           Tensor,
                                           Tensor]:

    """Implements Top2Gating on logits."""
    # everything is in fp32 in this function
    gates = F.softmax(logits, dim=1)

    capacity = _capacity(gates,
                         torch.tensor(capacity_factor * 2),
                         torch.tensor(min_capacity))

    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1)
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)

    # Create a mask for 2nd's expert per token using Gumbel-max trick
    # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
    logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    # Replace top-expert with min value
    logits_except1 = logits_w_noise.masked_fill(mask1.bool(), float("-inf"))
    indices2_s = torch.argmax(logits_except1, dim=1)
    mask2 = F.one_hot(indices2_s, num_classes=num_experts)

    # Compute locations in capacity buffer
    locations1 = torch.cumsum(mask1, dim=0) - 1
    locations2 = torch.cumsum(mask2, dim=0) - 1
    # Update 2nd's location by accounting for locations of 1st
    locations2 += torch.sum(mask1, dim=0, keepdim=True)

    # gating decisions
    exp_counts = torch.sum(mask1, dim=0).detach().to('cpu')

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.mean(me * ce) * num_experts * num_experts

    # Remove locations outside capacity from mask
    mask1 *= torch.lt(locations1, capacity)
    mask2 *= torch.lt(locations2, capacity)

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)

    # Normalize gate probabilities
    mask1_float = mask1.float()
    mask2_float = mask2.float()
    gates1_s = einsum("se,se->s", gates, mask1_float)
    gates2_s = einsum("se,se->s", gates, mask2_float)
    denom_s = gates1_s + gates2_s
    # Avoid divide-by-zero
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    gates1_s /= denom_s
    gates2_s /= denom_s

    # Calculate combine_weights and dispatch_mask
    gates1 = einsum("s,se->se", gates1_s, mask1_float)
    gates2 = einsum("s,se->se", gates2_s, mask2_float)
    locations1_sc = _one_hot_to_float(locations1_s, capacity)
    locations2_sc = _one_hot_to_float(locations2_s, capacity)
    combine1_sec = einsum("se,sc->sec", gates1, locations1_sc)
    combine2_sec = einsum("se,sc->sec", gates2, locations2_sc)
    combine_weights = combine1_sec + combine2_sec
    dispatch_mask = combine_weights.bool()

    return l_aux, combine_weights, dispatch_mask, exp_counts

class TopKGate(nn.Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::
        gate = TopKGate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)
    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf
    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """

    wg: torch.nn.Linear

    def __init__(self,
                 model_dim: int,
                 num_experts: int,
                 k: int = 1,
                 capacity_factor: float = 1.0,
                 eval_capacity_factor: float = 1.0,
                 min_capacity: int = 0,
                 noisy_gate_policy: Optional[str] = None,) -> None:
        super().__init__()

        # Only top-1 and top-2 are supported at the moment.
        if k != 1 and k != 2:
            raise ValueError('Only top-1 and top-2 gatings are supported.')
        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False).float()
        self.k = k
        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        self.min_capacity = min_capacity
        self.noisy_gate_policy = noisy_gate_policy

    def forward(
            self,
            input: torch.Tensor,) -> Tuple[Tensor,
                                           Tensor,
                                           Tensor,
                                           Tensor]:  # type: ignore


        if self.wg.weight.dtype != torch.float32:
            self.wg = self.wg.float()
        input_fp32 = input.float()

        # # input jittering
        # if self.noisy_gate_policy == 'Jitter' and self.training:
        #     input_fp32 = multiplicative_jitter(input_fp32, device=input.device)
        logits = self.wg(input_fp32)

        gate_output = top2gating(
            logits,
            self.capacity_factor if self.training else self.eval_capacity_factor,
            self.min_capacity)

        return gate_output

class MOELayer(nn.Module):
    """MOELayer module which implements MixtureOfExperts as described in Gshard_.
    ::
        gate = TopKGate(model_dim, num_experts)
        moe = MOELayer(gate, expert)
        output = moe(input)
        l_aux = moe.l_aux
    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf
    Args:
        gate (torch.nn.Module):
            gate network
        expert (torch.nn.Module):
            expert network
    """
    def __init__(self,
                 gate: nn.Module,
                 experts: nn.Module,
                 num_local_experts: int,) -> None:
        super().__init__()
        self.gate = gate
        self.experts = experts
        self.world_size = 1
        self.num_local_experts = num_local_experts

    def forward(self, input: Tensor,) -> Tensor:
        # Implement Algorithm 2 from GShard paper.
        d_model = input.shape[-1]

        # Initial implementation -> Reshape into S tokens by dropping sequence dimension.
        # Reshape into G groups so that each group can distribute tokens equally
        # group_size = kwargs['group_size'] if 'group_size' in kwargs.keys() else 1
        reshaped_input = input.reshape(-1, d_model)
        print(f"reshaped_input.size(): {reshaped_input.size()}")          

        self.l_aux, combine_weights, dispatch_mask, self.exp_counts = self.gate(reshaped_input)
        dispatched_input = einsum("sec,sm->ecm",
                                  dispatch_mask.type_as(input),
                                  reshaped_input)

        print(f"dispatched_input.size(): {dispatched_input.size()}")

        # Re-shape after all-to-all: ecm -> gecm
        dispatched_input = dispatched_input.reshape(self.world_size,
                                                    self.num_local_experts,
                                                    -1,
                                                    d_model)

        print(f"dispatched_input.size(): {dispatched_input.size()}")                                                    

        expert_output = self.experts(dispatched_input)
        print(f"expert_output.size(): {expert_output.size()}")                                                    
        exit()

        # Re-shape back: gecm -> ecm
        expert_output = expert_output.reshape(self.world_size * self.num_local_experts,
                                              -1,
                                              d_model)

        combined_output = einsum("sec,ecm->sm",
                                 combine_weights.type_as(input),
                                 expert_output)

        a = combined_output.reshape(input.shape)

        return a        



class Net(nn.Module):
    def __init__(self, num_experts, ):
        super().__init__()
        fc1 = nn.Linear(in_features=768, out_features=2) # in_features is static from ViT
        experts = Experts(fc1, num_local_experts=num_experts)
        self.moe = MOELayer(
            TopKGate(model_dim=768, num_experts=num_experts, k=2),
            experts,
            num_local_experts=num_experts
        )
        # self.fc2 = nn.Linear()


    def forward(self, x):
        output = self.moe(x)
        return output, self.moe.l_aux


if __name__ == "__main__":
    lr = 0.001
    n_class = 10
    criterion = nn.CrossEntropyLoss()

    train_embedding_path = "cifar10_train_embedding.pt"
    test_embedding_path = "cifar10_test_embedding.pt"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random_replay = RandomReplay(mem_size=3000)
    lwf = "lwf"

    # data, class_order = get_data(
    #     train_embedding_path, 
    #     test_embedding_path, 
    #     num_tasks=5, 
    #     validation=0.2,
    #     seed=42)    

    # print(len(data))
    # print(data[0]["train"]["x"][0].size())

    # gate = TopKGate(model_dim=768, num_experts=5, k=2)
    # # input_ = data[0]["train"]["x"][0]
    input_ = torch.rand(3, 768)
    # l_aux, combine_weights, dispatch_mask, exp_counts = gate(input_)
    # print(l_aux)
    # print(combine_weights)
    # print(dispatch_mask)
    # print(exp_counts)

    # fc = nn.Linear(in_features=768, out_features=2) 
    # experts = Experts(fc, num_local_experts=5)   
    # experts(input_)

    net = Net(num_experts=5)
    net(input_)
    print("all went well")