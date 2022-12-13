import copy
import itertools
import numpy as np
import random
import torch
import torchvision.models as models

from scipy.spatial.distance import cdist

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
            buffer_train_x, buffer_train_y = self._extend_dataset(dataset)
        elif self.buffer and uniform:
            self.cur_classes = dataset["classes"]
            self.classes = np.append(self.classes, self.cur_classes)            
            item_per_class = np.array([self.mem_size // len(self.classes)] * len(self.classes))
            for i in range(self.mem_size % len(self.classes)):
                item_per_class[i] += 1

            # update current buffer with all tasks' dataset
            buffer_train_x, buffer_train_y = self._extend_dataset(dataset)
            buffer = {"x": buffer_train_x, "y": buffer_train_y}
            self._sample_buffer(item_per_class, self.classes, buffer)
            buffer_train_x, buffer_train_y = self.buffer["x"], self.buffer["y"]
            
            # item_per_class = np.array([self.mem_size // len(self.classes)] * len(self.classes))
            # for i in range(self.mem_size % len(self.classes)):
            #     item_per_class[i] += 1

            # # update current buffer with all tasks' dataset
            # buffer_train_x, buffer_train_y = self._extend_dataset(dataset)
            # buffer = {"x": buffer_train_x, "y": buffer_train_y}
            # self._sample_buffer(item_per_class, self.classes, buffer)
            # buffer_train_x, buffer_train_y = self.buffer["x"], self.buffer["y"]
        else:
            # add current dataset to the buffer
            # only called once, i.e. after training the first task
            self.cur_classes = dataset["classes"]
            self.classes = dataset["classes"]
            buffer_train_x, buffer_train_y = dataset["train"]["x"], dataset["train"]["y"]

        self.buffer = {"x": buffer_train_x, "y": buffer_train_y, }
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
                y_idx = np.where(classes == y, )
                if item_per_class[y_idx] > 0:
                    item_per_class[y_idx] -= 1

                    buffer_x.append(x)
                    buffer_y.append(y)
            else:
                break

        self.buffer["x"] = buffer_x
        self.buffer["y"] = buffer_y

class Herding(Replay):
    def __init__(self, mem_size=3000, extractor=None):
        super().__init__(mem_size, extractor=extractor)

    # def update_buffer(self, dataset, uniform=False):
    #     """
    #     Before training current task, update the dataset buffer for the next task

    #     dataset: current task dataset
    #     """        
    #     # x_old, y_old = None, None

    #     if self.buffer:
    #         item_per_class = np.array([self.mem_size // self.n_class] * self.n_class)
    #         print(item_per_class)
    #         exit()
    #     else:
    #         # add current dataset to the buffer
    #         # only called once, i.e. after training the first task
    #         self.cur_classes = dataset["classes"]
    #         self.classes = dataset["classes"]
    #         buffer_x, buffer_y = self._get_buffer(dataset)

    #     # self.buffer = {"x": buffer_x, "y": buffer_y, "x_old": x_old, "y_old": y_old}
    #     self.buffer = {"x": buffer_x, "y": buffer_y}
    #     self.n_class = len(self.classes)

    def _get_buffer(self, dataset):
        buffer_x = copy.deepcopy(dataset["train"]["x"])
        buffer_y = copy.deepcopy(dataset["train"]["y"])
        val = dataset.get("val", [])
        if val:
            buffer_x += val["x"]
            buffer_y += val["y"]
        return buffer_x, buffer_y

    def store_buffer(self, dataset):    
        if self.buffer:
            # update old buffer with new dataset
            # take the top k herding, where k = memory // n.class
            self.cur_classes = dataset["classes"]
            old_classes = self.classes
            self.classes = np.append(old_classes, self.cur_classes)
            item_per_class = np.array([self.mem_size // len(self.classes)] * len(self.classes))
            print(f"\tITEM_PER_CLASS: {item_per_class}")

            old_label_feature = self._make_label_feature_dictionary()
            new_x, new_y = self._get_buffer(dataset)
            new_label_feature = self._make_label_feature_dictionary(x=new_x, y=new_y)

            x, y = self._extend_herding(np.unique(old_classes), old_label_feature, item_per_class[0])
            x_, y_ = self._extend_herding(np.unique(self.cur_classes), new_label_feature, item_per_class[0])
            x.extend(x_)
            y.extend(y_)
            self.buffer = {"x": x, "y": y}
        else:
            self.cur_classes = dataset["classes"]
            self.classes = self.cur_classes
            item_per_class = np.array([self.mem_size // len(self.classes)] * len(self.classes))
            print(f"\tITEM_PER_CLASS: {item_per_class}")
            new_x, new_y = self._get_buffer(dataset)
            new_label_feature = self._make_label_feature_dictionary(x=new_x, y=new_y)
            x, y = self._extend_herding(np.unique(self.cur_classes), new_label_feature, item_per_class[0])          
            self.buffer = {"x": x, "y": y}
        

    def _extend_herding(self, labels, label_feature, m):
        x, y = [], []
        print(f"Herding {m} exemplars for each {labels}")
        for label in labels:
            pos = torch.tensor(self._herding(label_feature[label].detach(), m))
            x_ = torch.index_select(label_feature[label], 0, pos)
            y_ = [label for _ in range(len(x_))]         
            x.extend(x_)
            y.extend(y_)
        return x, y

    def _split_train_val(self, dataset):
        """
        Separate the given dataset into train/val using 9/1 split.
        Also split buffer using 9/1 if exists.
        """ 
        train_x, train_y, val_x, val_y = [], [], [], []
        # split buffer first if exist
        ipc_t, ipc_v = None, None
        if self.buffer:
            t_x, t_y, v_x, v_y, ipc_t, ipc_v = self._split(self.buffer, None)
            train_x.extend(t_x)
            train_y.extend(t_y)
            val_x.extend(v_x)
            val_y.extend(v_y)

        new_x, new_y = self._get_buffer(dataset)
        t_x, t_y, v_x, v_y, _, _ = self._split({"x": new_x, "y": new_y}, None, ipc_v=ipc_v)
        train_x.extend(t_x)
        train_y.extend(t_y)
        val_x.extend(v_x)
        val_y.extend(v_y)

        return train_x, train_y, val_x, val_y

    def _get_item_per_class(self, population, n_class):
        return np.array([population // n_class] * n_class)

    def _split(self, dataset, label_feature, ipc_t=None, ipc_v=None):
        pop_total = len(dataset["x"])
        pop_val = int(pop_total * 0.1)
        pop_train = int(pop_total * 0.9)            
        
        n_class = len(np.unique(dataset["y"]))
        item_per_class_train = self._get_item_per_class(pop_train, n_class)
        item_per_class_val = self._get_item_per_class(pop_val, n_class)
        train_x, train_y, val_x, val_y = [], [], [], []
        y_arr = np.array(dataset["y"])
        x_tensor = torch.stack(dataset["x"], 0)
        for label, (ipc_train, ipc_val) in zip(np.unique(dataset["y"]), zip(item_per_class_train, item_per_class_val)): 
            # get instances where y == label
            y_idx = np.where(y_arr == label)[0]
            x_label = x_tensor[y_idx]
            total = ipc_train + ipc_val
            t_x = x_label[:ipc_train]
            v_x = x_label[ipc_train:total]
            t_y = [label for _ in range(len(t_x))]                     
            v_y = [label for _ in range(len(v_x))]

            train_x.extend(t_x)
            train_y.extend(t_y)            
            val_x.extend(v_x[:ipc_v] if ipc_v else v_x)
            val_y.extend(v_y[:ipc_v] if ipc_v else v_y)
        return train_x, train_y, val_x, val_y, item_per_class_train[0], item_per_class_val[0]

    def _extend_dataset(self, dataset):
        x_extended = self.buffer["x_train"] + dataset["train"]["x"]
        y_extended = self.buffer["y_train"] + dataset["train"]["y"]
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
                y_idx = np.where(classes == y, )
                if item_per_class[y_idx] > 0:
                    item_per_class[y_idx] -= 1

                    buffer_x.append(x)
                    buffer_y.append(y)
            else:
                break

        self.buffer["x"] = buffer_x
        self.buffer["y"] = buffer_y

    def _make_label_feature_dictionary(self, x=None, y=None):
        """
        Create a dictionary where key is class label and value is features.
        This will be used when herding the top m samples for the given label.
        """
        if x == None:
            x = self.buffer["x"]
        if y == None:
            y = self.buffer["y"]

        label_feature = {}
        train_labels = np.unique(y)        
        train_tensor = torch.stack(x)
        # print(train_labels)
        for label in train_labels:
            idx = np.argwhere(y == label).reshape(-1)
            label_feature[label] = train_tensor[idx]
        # print("finish creating label_feature")
        return label_feature

    def _herding(self, features, nb_exemplars):
        # adapted from avalanche construct_exemplar_set()
        # https://github.com/ContinualAI/avalanche/blob/master/avalanche/training/supervised/icarl.py

        D = features.T
        D = D / torch.norm(D, dim=0)
        mu = torch.mean(D, dim=1)
        order = torch.zeros(features.size(0))
        w_t = mu
        i, added, selected = 0, 0, []
        while not added == nb_exemplars and i < 1000:
            tmp_t = torch.mm(w_t.unsqueeze(0), D)
            ind_max = torch.argmax(tmp_t)
            if ind_max not in selected:
                order[ind_max] = 1 + added
                added += 1
                selected.append(ind_max.item())
            w_t = w_t + mu - D[:, ind_max]
            i += 1
        return selected

    def compute_class_means(self):
        # adapted from avalanche compute_class_mean()
        # https://github.com/ContinualAI/avalanche/blob/master/avalanche/training/supervised/icarl.py
        
        label_feature = self._make_label_feature_dictionary()
        n_classes = len(label_feature)
        class_means = torch.zeros((768, n_classes))
        self.cls2idx = {c: i for i, c in enumerate(label_feature.keys())}
        self.idx2cls = {i: c for i, c in enumerate(label_feature.keys())}

        for label, features in label_feature.items():
            D = features.T
            D = D / torch.norm(D, dim=0)

            D2 = features.T
            D2 = D2 / torch.norm(D2, dim=0)

            div = torch.ones(features.size(0))
            div = div / features.size(0)

            m1 = torch.mm(D, div.unsqueeze(1)).squeeze(1)
            m2 = torch.mm(D2, div.unsqueeze(1)).squeeze(1)

            class_means[:, self.cls2idx[label]] = (m1 + m2) / 2
            class_means[:, self.cls2idx[label]] /= torch.norm(class_means[:, self.cls2idx[label]])

        self.class_means = class_means.T
        

    # def compute_accuracy(model, loader, class_means):
    #     features, targets_ = utils.extract_features(model, loader)

    #     features = (features.T / (np.linalg.norm(features.T, axis=0) + EPSILON)).T

    #     # Compute score for iCaRL
    #     sqd = cdist(class_means, features, 'sqeuclidean')
    #     score_icarl = (-sqd).T

    #     return score_icarl, targets_        

if __name__ == "__main__":
    train_embedding_path = "cifar100_train_embedding.pt"
    test_embedding_path = "cifar100_test_embedding.pt"
    val_embedding_path = None

    # train_embedding_path = "imagenet1000_train_embedding.pt"
    # test_embedding_path = None
    # val_embedding_path = "imagenet1000_val_embedding.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    replay = Herding(mem_size=500)
    data, class_order = get_data(
        train_embedding_path, 
        test_embedding_path, 
        val_embedding_path=val_embedding_path,
        num_tasks=20,
        validation=0.2,
    )   
    print("finish populating data")
    print(class_order)

    # x = data[0]["train"]["x"]
    # # grah = replay._herding(x, 10)
    # grah = torch.stack(x)
    # grah = grah.detach().cpu().numpy()
    # print(grah.shape)
    # pos = replay._herding(grah, 10)
    # print(pos)

    # replay.update_buffer(data[0])
    # label_feature = replay._make_label_feature_dictionary()

    for i in range(len(data)):
        print(i+1)
        x_train, y_train, x_val, y_val = replay._split_train_val(data[i])
        replay.store_buffer(data[i])
        replay.compute_class_means()
        print(replay.class_means.size())
        print(replay.idx2cls)
        print()
        print()

    print(replay.class_means.si)


    # print()
    # label_feature = replay._make_label_feature_dictionary()
    # for k, v in label_feature.items():
    #     print(k, v.size())
    # print(len(label_feature.keys()))
    # new_x, new_y = replay._get_buffer(data[1])
    # label_feature_n = replay._make_label_feature_dictionary(x=new_x, y=new_y)
    # replay._split_train_val(label_feature=label_feature)

    # replay.update_buffer(data[1])
    # replay.store_buffer(data[0])
    # replay.store_buffer(data[1])
    # replay.store_buffer(data[2])
    # replay.store_buffer(data[3])
    # replay.store_buffer(data[4])
    # replay.store_buffer(data[5])

    # print(data[3])
    