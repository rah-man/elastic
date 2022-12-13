import numpy as np
import sys
import torch

from base import get_data
from replay import RandomReplay
from utils import parse_input

if __name__ == "__main__":
    # STEPS:
    # 1. load the task datasets (as usual)
    # 2. set the hyperparameters
    # 3. pass the datasets and hyperparams to bic trainer
    # 4. BiC trainer does:
    #   a. two steps training
    #   b. icarl herding?
    #   c. ??

    train_embedding_path = "cifar100_train_embedding.pt"
    test_embedding_path = "cifar100_test_embedding.pt"
    val_embedding_path = None

    batch, n_class, steps, mem_size, epochs, n_experts, k = parse_input(sys.argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random_replay = RandomReplay(mem_size=mem_size)
    data, class_order = get_data(
        train_embedding_path, 
        test_embedding_path, 
        val_embedding_path=val_embedding_path,
        num_tasks=n_experts,
        validation=0.2,
    )   

    print(class_order)