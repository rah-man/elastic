import copy
import numpy as np
import torch
import torchvision.models as models
import random

from collections import Counter
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import datasets, transforms
from torchvision.models.feature_extraction import create_feature_extractor


class BaseDataset(Dataset):
    def __init__(self, x, y, transform=None, cls2idx=None):
        self.images = x
        self.labels = y
        self.transform = transform
        self.cls2idx = cls2idx

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        x = self.images[index]
        y = self.labels[index]

        if self.transform:
            x = self.transform(x)
        if self.cls2idx:
            y = self.cls2idx[y]
        
        return x, y


def get_data(
    train_embedding_path, 
    test_embedding_path, 
    val_embedding_path=None,
    num_tasks=5, 
    classes_in_first_task=None, 
    validation=0.0, 
    shuffle_classes=True, 
    k=2, 
    transform=None, 
    dummy=False,
    seed=None):
    """
    # torch_dataset: one instance of torchvision datasets --> removed as using embedding input
    # data_path: path to store the dataset --> removed as now it needs two specific embedding input paths

    Using new data input, i.e. not raw image pixels but vision transformer's embedding, the input structure provided by
    train_embedding_path and test_embedding_path are like this:
    data = {
        'data': [
            [
                768_dimension_of_ViT_embedding,
                the_label
            ],
            [],
            ...
        ],
        'targets': labels/classes as a whole
    }    

    train_embedding_path: the path to ViT embedding train file
    test_embeding_path: the path to ViT embedding test file
    num_tasks: the number of tasks, this may be ignored if classes_in_first_task is not None
    classes_in_first_task: the number of classes in the first task. If None, the classes are divided evenly per task
    validation: floating number for validation size (e.g. 0.20)
    shuffle_classes: True/False to shuffle the class order
    k: the number of classes in the remaining tasks (only used if classes_in_first_task is not None)

    # transform : image transformer object. Use ViT weight transform. --> removed as the input is already in embedding format

    dummy: set to True to get only a small amount of data (for small testing on CPU)

    return:
    data: a dictionary of dataset for each task
        data = {
            [{
                'name': task-0,
                'train': {
                    'x': [],
                    'y': []
                },
                'val': {
                    'x': [],
                    'y': []
                },
                'test': {
                    'x': [],
                    'y': []
                },
                'classes': int
            }],
        }
    class_order: the order of the classes in the dataset (may be shuffled)
    """

    """

    """

    data = {}

    # trainset = torch_dataset(root=data_path, train=True, download=True, transform=transform)
    # testset = torch_dataset(root=data_path, train=False, download=True, transform=transform)

    trainset = torch.load(train_embedding_path)
    if test_embedding_path:
        testset = torch.load(test_embedding_path)
    if val_embedding_path:
        valset = torch.load(val_embedding_path)

    num_classes = len(np.unique(trainset["targets"]))
    # else:
    #     # this is for imagenet as there's no "targets"
    #     classes = trainset["labels"]
    #     classes = torch.stack(classes)
    #     num_classes = len(np.unique(classes))
    #     imagenet = True
    class_order = list(range(num_classes))
    
    if shuffle_classes:
        if seed:
            np.random.seed(seed)
        np.random.shuffle(class_order)

    if classes_in_first_task is None:
        # Divide evenly the number of classes for each task
        cpertask = np.array([num_classes // num_tasks] * num_tasks)
        for i in range(num_classes % num_tasks):
            cpertask[i] += 1
    else:
        # Allocate the rest of the classes based on k
        remaining_classes = num_classes - classes_in_first_task
        cresttask = remaining_classes // k
        cpertask = np.array([classes_in_first_task] + [remaining_classes // cresttask] * cresttask)
        for i in range(remaining_classes % k):
            cpertask[i + 1] += 1
        num_tasks = len(cpertask)

    cpertask_cumsum = np.cumsum(cpertask)

    total_task = num_tasks
    for tt in range(total_task):
        data[tt] = {}
        data[tt]['name'] = 'task-' + str(tt)
        data[tt]['train'] = {'x': [], 'y': []}
        data[tt]['val'] = {'x': [], 'y': []}
        data[tt]['test'] = {'x': [], 'y': []}
        # data[tt]['nclass'] = cpertask[tt]

    # Populate the train set
    # for i, (this_image, this_label) in enumerate(trainset):
    for i, (this_image, this_label) in enumerate(trainset["data"]):
        original_label = int(this_label)
        this_label = class_order.index(original_label)
        this_task = (this_label >= cpertask_cumsum).sum()

        data[this_task]['train']['x'].append(this_image)
        data[this_task]['train']['y'].append(original_label)

        if dummy and i >= 500:
            break

    # Populate the test set
    # for i, (this_image, this_label) in enumerate(testset):
    if test_embedding_path:
        for i, (this_image, this_label) in enumerate(testset["data"]):
            original_label = int(this_label)
            this_label = class_order.index(original_label)
            this_task = (this_label >= cpertask_cumsum).sum()

            data[this_task]['test']['x'].append(this_image)
            data[this_task]['test']['y'].append(original_label)

            if dummy and i >= 100:
                break

    # Populate validation if required
    if val_embedding_path:
        # if there's a special validation set, i.e. ImageNet
        for i, (this_image, this_label) in enumerate(valset["data"]):
            original_label = int(this_label)
            this_label = class_order.index(original_label)
            this_task = (this_label >= cpertask_cumsum).sum()

            data[this_task]['val']['x'].append(this_image)
            data[this_task]['val']['y'].append(original_label)       
    elif validation > 0.0:
        for tt in data.keys():
            pop_idx = [i for i in range(len(data[tt]["train"]["x"]))]
            val_idx = random.sample(pop_idx, int(np.round(len(pop_idx) * validation)))
            val_idx.sort(reverse=True)

            for ii in range(len(val_idx)):
                data[tt]['val']['x'].append(data[tt]['train']['x'][val_idx[ii]])
                data[tt]['val']['y'].append(data[tt]['train']['y'][val_idx[ii]])
                data[tt]['train']['x'].pop(val_idx[ii])
                data[tt]['train']['y'].pop(val_idx[ii])     

    for tt in range(total_task):
        data[tt]["classes"] = np.unique(data[tt]["train"]["y"])


    return data, class_order

class Extractor:
    """
    Create a feature extractor wrapper (e.g. ViT) to generate image embedding.

    NOTE: may not be used anymore as the data input is already embedded
    """
    def __init__(self, model_, weights=None, return_nodes=None, device="cpu"):
        self.model = model_(weights=weights)
        self.weights = weights
        self.return_nodes = return_nodes
        self.feature_extractor = create_feature_extractor(self.model, return_nodes=return_nodes).to(device)
        self.feature_extractor.eval()

    def __call__(self, x):
        """
        Assuming there is only one final layer where the value needs to be extracted, therefore use index 0
        Using Vision Transformer (ViT), the last layer before the classification layer is 'getitem_5'
        """
        with torch.no_grad():
            extracted = self.feature_extractor(x)
        return extracted.get(self.return_nodes[0])

# def get_extractor(model_, weights=None, return_nodes=None):
#     model = model_(weights=weights)
#     print(f"CREATING EXTRACTOR: {return_nodes}")
#     feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)

#     return feature_extractor

def extract_once(datapath, torch_dataset, transform, extractor, device, train=True, outfile=None):
    dataset = torch_dataset(root=datapath, train=train, download=True, transform=transform)
    targets = dataset.targets

    embedding = []
    for i, (image, label) in enumerate(dataset):
        image = torch.unsqueeze(image, 0).to(device)
        extracted = torch.squeeze(extractor(image)).cpu()
        embedding.append([extracted, label])
        
    extraction = {"data": embedding, "targets": targets}
    torch.save(extraction, outfile)

cifar100_coarse_labels = [4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
                          3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                          6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
                          0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
                          5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
                          16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
                          10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
                          2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
                          16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
                          18,  1,  2, 15,  6,  0, 17,  8, 14, 13]

cifar100_coarse_idx = {k:v for k, v in enumerate(cifar100_coarse_labels)}    

def extract_once_coarse(datapath, torch_dataset, transform, extractor, device, train=True, outfile=None):
    coarse_label = np.array(cifar100_coarse_labels)
    dataset = torch_dataset(root=datapath, train=train, download=True, transform=transform)
    targets = coarse_label[dataset.targets].tolist()

    embedding = []
    for i, (image, label) in enumerate(dataset):
        image = torch.unsqueeze(image, 0).to(device)
        extracted = torch.squeeze(extractor(image)).cpu()
        embedding.append([extracted, cifar100_coarse_labels[label]])
        
    extraction = {"data": embedding, "targets": targets}
    torch.save(extraction, outfile)

def extract_imagenet(path, vit_transform, extractor, device, batch_size=256, outfile=None):
    dataset = datasets.ImageFolder(path, transform=vit_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    

    embeddings, targets = [], []

    for i, (images, labels_) in enumerate(dataloader):
        images = images.to(device)
        extracted = extractor(images.to(device))
        e = extracted.detach().cpu()
        l = labels_.detach().cpu().numpy().tolist()
        embeddings.extend(zip(e, l))
        targets.extend(l)
        print(np.unique(labels_))
        torch.cuda.empty_cache()
        # break

    # NOTE: "data" should be in a pair of (embeddings, label)
    # NOTE: "labels" should not be there
    # NOTE: "targets" should store all labels in their particular order
    extraction = {"data": embeddings, "targets": targets}
    torch.save(extraction, outfile)


if __name__ == "__main__":
    train_embedding_path = "imagenet1000_train_embedding.pt"
    test_embedding_path = None
    val_embedding_path = "imagenet1000_val_embedding.pt"
    n_experts = 20

    data, class_order = get_data(
        train_embedding_path, 
        test_embedding_path, 
        val_embedding_path=val_embedding_path,
        num_tasks=n_experts, # num_tasks == n_experts
        validation=0.2,)

    exit()
    #############################################

    # inp = "imagenet100_train_embedding.pt"
    # extraction = torch.load(inp)
    # print(extraction.keys())

    # data = extraction["data"]
    # labels = extraction["labels"]

    # print(len(data))
    # print("\t", data[0].size())
    # print(len(labels))
    # print("\t", labels[-1])

    # exit()

    # print(len(emb["data"]))
    # embeddings = emb["data"][0]
    # labels = emb["data"][1]

    # print(labels)

    # exit()
    #########################################

    model_ = models.vit_b_16
    weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
    vit_transform = weights.transforms()
    return_nodes = ["getitem_5"]
    device = "cuda" if torch.cuda.is_available() else "cpu"    

    # path = "../Documents/imagenet-100/train/"
    # outfile = "imagenet100_train_embedding.pt"

    path = "../Documents/imagenet/val/"
    outfile = "imagenet1000_val_embedding.pt"

    extractor = Extractor(model_, weights=weights, return_nodes=return_nodes, device=device)
    extract_imagenet(path, vit_transform, extractor, device, batch_size=256, outfile=outfile)    

    path = "../Documents/imagenet/train/"
    outfile = "imagenet1000_train_embedding.pt"

    extractor = Extractor(model_, weights=weights, return_nodes=return_nodes, device=device)
    extract_imagenet(path, vit_transform, extractor, device, batch_size=256, outfile=outfile)

    exit()
    #################################################

    # torch_dataset = datasets.CIFAR10
    # datapath = "../CIFAR_data/"
    # model_ = models.vit_b_16
    # weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
    # transform = weights.transforms()
    # return_nodes = ["getitem_5"]
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # extractor = Extractor(model_, weights=weights, return_nodes=return_nodes, device=device)
    # extract_once(datapath, torch_dataset, transform, extractor, device, train=True, outfile="train_embedding.pt")

    # data, class_order = get_data(
    #     torch_dataset, data_path, num_tasks=5, shuffle_classes=True, classes_in_first_task=None, k=2, validation=0.2, dummy=True)

    # for k, v in data.items():
    #     print(f"{v['name']}\n\tlen(train): {len(v['train']['x'])}\n\tlen(val): {len(v['val']['x'])}\n\tlen(test): {len(v['test']['x'])}\n\tclasses: {v['classes']}\n\tnclass: {len(v['classes'])}")

    # print(class_order)
    # cls2id = {v: k for k, v in enumerate(class_order)}
    # id2cls = {k: v for k, v in enumerate(class_order)}
    # print(cls2id)
    # print(id2cls)

    # imgarr = np.asarray(data[0]["train"]["x"][0])
    # print(data[0]["train"]["x"][0])
    # print(data[0]["train"]["y"][0])
    # print(imgarr.shape)

    ###################################################

    # transform CIFAR100
    datapath = "CIFAR_data"
    torch_dataset = datasets.CIFAR100
    model_ = models.vit_b_16
    weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
    transform = weights.transforms()
    return_nodes = ["getitem_5"]
    device = "cuda" if torch.cuda.is_available() else "cpu"    

    extractor = Extractor(model_, weights=weights, return_nodes=return_nodes, device=device)
    extract_once_coarse(datapath, torch_dataset, transform, extractor, device, train=True, outfile="cifar100_coarse_train_embedding.pt")
    extract_once_coarse(datapath, torch_dataset, transform, extractor, device, train=False, outfile="cifar100_coarse_test_embedding.pt")


    ###################################################

    # train_embedding_path = "cifar10_train_embedding.pt"
    # test_embedding_path = "cifar10_test_embedding.pt"

    # data, class_order = get_data(
    #     train_embedding_path, 
    #     test_embedding_path, 
    #     num_tasks=5, 
    #     validation=0.2,)

    # cls2id = {v: k for k, v in enumerate(class_order)}

    # print(class_order)
    # print(cls2id)
    # for k, v in data.items():
    #     print(f"{v['name']}\n\tlen(train): {len(v['train']['x'])}\n\tlen(val): {len(v['val']['x'])}\n\tlen(test): {len(v['test']['x'])}\n\tclasses: {v['classes']}\n\tnclass: {len(v['classes'])}")


