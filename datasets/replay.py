import numpy as np
import torch
import torchvision.models as models

from base import BaseDataset, Extractor, get_data
from collections import Counter
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import datasets
from torchvision.models.feature_extraction import create_feature_extractor

class Replay:
    def __init__(self, mem_size, extractor=None):
        """
        Question: should the buffer contains the extracted values or the raw values? (for herding etc)
        """
        self.mem_size = mem_size
        self.extractor = extractor
        self.buffer = None
        self.n_class = 0
        self.classes = None
        self.cur_classes = None

    def update_buffer(self, dataset):
        pass


class RandomReplay(Replay):
    def __init__(self, mem_size=3000, extractor=None):
        super().__init__(mem_size, extractor=extractor)

    def update_buffer(self, dataset, uniform=False):
        """
        Before training current task, update the dataset buffer for the next task

        dataset: current task dataset
        """        
        if self.buffer and not uniform:
            item_per_class = np.array([self.mem_size // self.n_class] * self.n_class)            
            for i in range(self.mem_size % self.n_class):
                item_per_class[i] += 1
            
            # update current buffer with previous task's dataset
            self._sample_buffer(item_per_class, self.classes, self.buffer)
            self.cur_classes = dataset["classes"]
            self.classes = np.append(self.classes, self.cur_classes)
            buffer_x, buffer_y = self._extend_dataset(dataset)
        elif self.buffer and uniform:
            self.cur_classes = dataset["classes"]
            self.classes = np.append(self.classes, self.cur_classes)            
            item_per_class = np.array([self.mem_size // len(self.classes)] * len(self.classes))
            for i in range(self.mem_size % len(self.classes)):
                item_per_class[i] += 1

            # update current buffer with all tasks' dataset
            buffer_x, buffer_y = self._extend_dataset(dataset)
            buffer = {"x": buffer_x, "y": buffer_y}
            self._sample_buffer(item_per_class, self.classes, buffer)
            buffer_x, buffer_y = self.buffer["x"], self.buffer["y"]
        else:
            # add current dataset to the buffer
            # only called once, i.e. after training the first task
            self.cur_classes = dataset["classes"]
            self.classes = dataset["classes"]
            buffer_x, buffer_y = dataset["train"]["x"], dataset["train"]["y"]
        
        self.buffer = {"x": buffer_x, "y": buffer_y}
        self.n_class = len(self.classes)

    def _extend_dataset(self, dataset):
        x_extended = self.buffer["x"] + dataset["train"]["x"]
        y_extended = self.buffer["y"] + dataset["train"]["y"]
        return x_extended, y_extended

    def _sample_buffer(self, item_per_class, classes, dataset):
        """
        Randomly select the instances to be kept in the next training

        item_per_class: a list of the number of items per class
        classes: a list of class (to get the index)
        dataset: a dictionary containing the training dataset or the combination of the buffer and the training dataset
            dataset{
                "x": [],
                "y": []
            }
        """
        sample_idx = np.random.permutation(np.arange(len(dataset["x"])))
        buffer_x, buffer_y = [], []
        for i in sample_idx:
            if not all(el == 0 for el in item_per_class):
                x, y = dataset["x"][i], dataset["y"][i]
                y_idx = np.where(classes == y)
                if item_per_class[y_idx] > 0:
                    item_per_class[y_idx] -= 1

                    buffer_x.append(x)
                    buffer_y.append(y)
            else:
                break

        self.buffer["x"] = buffer_x
        self.buffer["y"] = buffer_y

class iCARL(Replay):
    def __init__(self, mem_size):
        super().__init__(mem_size)

    def prepare_buffer(self, dataset):
        if not self.buffer:
            return dataset
        else:
            return ConcatDataset([BaseDataset(self.buffer), dataset["train"]])

    def update_buffer(self, dataset):
        pass

    def _herding(self, x, m, mean_=None):
        pos_s = []
        comb = 0
        mu = np.mean(x, axis=0, keepdims=False) if mean_ is None else mean_
        for k in range(m):
            det = mu * (k + 1) - comb
            dist = np.zeros(shape=(np.shape(x)[0]))
            for i in range(np.shape(x)[0]):
                if i in pos_s:
                    dist[i] = np.inf
                else:
                    dist[i] = np.linalg.norm(det - x[i])
            pos = np.argmin(dist)
            pos_s.append(pos)
            comb += x[pos]

        return pos_s    

if __name__ == "__main__":
    dataset = datasets.CIFAR10
    data_path = "../CIFAR_data/"
    model_ = models.vit_b_16
    weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
    return_nodes = ["getitem_5"]

    data, class_order = get_data(
        dataset, data_path, num_tasks=5, shuffle_classes=True, classes_in_first_task=None, k=2, validation=0.2)

    for k, v in data.items():
        print(f"{v['name']}\n\tlen(train): {len(v['train']['x'])}\n\tlen(val): {len(v['val']['x'])}\n\tlen(test): {len(v['test']['x'])}\n\tclasses: {v['classes']}\n\tnclass: {len(v['classes'])}")

    print(class_order)


    trainset = BaseDataset(data[0]["train"]["x"], data[0]["train"]["y"], transform=weights.transforms())
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    extractor = Extractor(model_, weights=weights, return_nodes=return_nodes)

    x, y = next(iter(trainloader))
    extractor(x)[-1]
    exit()

    random_replay = RandomReplay(mem_size=3000, extractor=extractor)

    for k, v in data.items():
        random_replay.update_buffer(v)
        print(len(random_replay.buffer["x"]), len(random_replay.buffer["y"]))
        print(Counter(random_replay.buffer["y"]))
        print()