# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py


import copy
import numpy as np
import torch
import torch.nn as nn

from torch.distributions.normal import Normal

class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates, kind="original"):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)
        self.kind = kind

    def dispatch(self, inp, labels):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        inp_split = torch.split(inp_exp, self._part_sizes, dim=0)
        
        lab_exp = labels[self._batch_index]
        lab_split = torch.split(lab_exp, self._part_sizes, dim=0)
        return inp_split, lab_split

    def combine(self, expert_out, expert_labels=None, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
             
        if self.kind == "dynamic":
            out_size_ = expert_out[-1].size(1)
            zeros = torch.zeros(self._gates.size(0), out_size_, requires_grad=True, device=stitched.device)            
            combined = zeros.index_add(0, self._batch_index, stitched)

        combined[combined == 0] = np.finfo(float).eps
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

class MLPDynamic(nn.Module):
    def __init__(self, input_size=768, hidden_size=256, output_size=2, projected_output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=output_size)        
        self.mapper = nn.Linear(in_features=output_size, out_features=projected_output_size, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.mapper(self.fc2(out))
        return out

class DynamycMoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, input_size=768, hidden_size=256, noisy_gating=True, k=4, kind="dynamic"):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.noisy_gating = noisy_gating
        self.k = k
        self.kind = kind
        self.experts = None
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))                
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.ce_loss = nn.CrossEntropyLoss()

    def expand_expert(self, seen_cls, new_cls, k=2):
        self.seen_cls = seen_cls
        self.new_cls = new_cls

        if not self.experts:
            # only executed once for the first task
            experts = nn.ModuleList([MLPDynamic(input_size=self.input_size, hidden_size=self.hidden_size, output_size=new_cls, projected_output_size=new_cls)])
            self.num_experts = len(experts)

            w_gate = nn.Parameter(torch.zeros(self.input_size, self.num_experts), requires_grad=True)
            w_noise = nn.Parameter(torch.zeros(self.input_size, self.num_experts), requires_grad=True)            
        else:
            # UPDATE THE MAPPER LAYER FOR EACH OF THE PREVIOUS EXPERT
            # 1. old_mapper.out_features = old_mapper.out_features + self.new_cls
            # 2. update zero weights for corresponding experts\

            experts = copy.deepcopy(self.experts)
            experts.append(MLPDynamic(input_size=self.input_size, hidden_size=self.hidden_size, output_size=new_cls, projected_output_size=new_cls))
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

            w_gate = nn.Parameter(torch.zeros(self.input_size, self.num_experts), requires_grad=True)
            w_noise = nn.Parameter(torch.zeros(self.input_size, self.num_experts), requires_grad=True)
            
            with torch.no_grad():
                w_gate.data[:, :self.w_gate.size(1)] = self.w_gate
                w_noise.data[:, :self.w_gate.size(1)] = self.w_noise

        
        self.k = min(k, self.num_experts) if k > self.num_experts else k        
        self.w_gate = w_gate
        self.w_noise = w_noise
        self.experts = experts
    
    def freeze_previous(self):
        for i in range(len(self.experts) - 1):
            e = self.experts[i]
            for param in e.parameters():
                param.requires_grad = False

    def freeze_all(self):
        for e in self.experts:
            for param in e.parameters():
                param.requires_grad = False

    def unfreeze_all(self):
        for e in self.experts:
            for param in e.parameters():
                param.requires_grad = True
        
    def print_mapper_weights(self):        
        if self.experts:
            print("k =", self.k)
            for expert in self.experts:
                print(expert.mapper.weight)
                print()

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, labels, loss_coef=1e-2):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        gates, load = self.noisy_top_k_gating(x, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        #
        gate_loss = self.cv_squared(importance) + self.cv_squared(load)
        gate_loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates, self.kind)
        expert_inputs, expert_labels = dispatcher.dispatch(x, labels)
        # gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        # expert_losses = dispatcher.calculate_expert_loss(expert_outputs, expert_labels)
        expert_losses = []
        for eo, el in zip(expert_outputs, expert_labels):
            expert_losses.append(self.ce_loss(eo, el))

        y = dispatcher.combine(expert_outputs, expert_labels)
        return y, gate_loss, expert_losses, gates

if __name__ == "__main__":
    moe = DynamycMoE(k=2)
    moe.print_mapper_weights()

    moe.expand_expert(0, 2, k=2)
    moe.print_mapper_weights()

    moe.expand_expert(2, 2, k=2)
    moe.print_mapper_weights()

    moe.expand_expert(4, 2, k=2)
    moe.print_mapper_weights()

    moe.expand_expert(6, 2, k=2)
    moe.print_mapper_weights()    

    moe.expand_expert(8, 2, k=2)
    moe.print_mapper_weights()        