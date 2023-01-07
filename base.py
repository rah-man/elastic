import copy
import numpy as np
import os
import pickle
import random
import scipy
import time
import torch
import torchvision.models as models

from collections import Counter
from PIL import Image
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
    seed=42,
    expert=False):
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
    taskcla = []

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
        # if seed:
        #     np.random.seed(seed)
        np.random.shuffle(class_order)
    print("CLASS_ORDER:", class_order)

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
    init_class = np.concatenate(([0], cpertask_cumsum[:-1]))

    total_task = num_tasks
    for tt in range(total_task):
        data[tt] = {}
        data[tt]['name'] = 'task-' + str(tt)
        data[tt]['trn'] = {'x': [], 'y': []}
        data[tt]['val'] = {'x': [], 'y': []}
        data[tt]['tst'] = {'x': [], 'y': []}
        # data[tt]['nclass'] = cpertask[tt]


    # Populate the train set
    # for i, (this_image, this_label) in enumerate(trainset):
    for i, (this_image, this_label) in enumerate(trainset["data"]):
        original_label = int(this_label)
        this_label = class_order.index(original_label)
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['trn']['x'].append(this_image)
        if not expert:
            data[this_task]['trn']['y'].append(this_label - init_class[this_task])
        else:
            data[this_task]['trn']['y'].append(original_label)

        if dummy and i >= 500:
            break

    # Populate the test set
    # for i, (this_image, this_label) in enumerate(testset):
    if test_embedding_path:
        for i, (this_image, this_label) in enumerate(testset["data"]):
            original_label = int(this_label)
            this_label = class_order.index(original_label)
            this_task = (this_label >= cpertask_cumsum).sum()

            data[this_task]['tst']['x'].append(this_image)
            if not expert:
                data[this_task]['tst']['y'].append(this_label - init_class[this_task])
            else:
                data[this_task]['tst']['y'].append(original_label)

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
            if not expert:
                data[this_task]['val']['y'].append(this_label - init_class[this_task])       
            else:
                data[this_task]['val']['y'].append(original_label)       
    elif validation > 0.0:
        for tt in data.keys():
            pop_idx = [i for i in range(len(data[tt]["trn"]["x"]))]
            val_idx = random.sample(pop_idx, int(np.round(len(pop_idx) * validation)))
            val_idx.sort(reverse=True)

            for ii in range(len(val_idx)):
                data[tt]['val']['x'].append(data[tt]['trn']['x'][val_idx[ii]])
                data[tt]['val']['y'].append(data[tt]['trn']['y'][val_idx[ii]])
                data[tt]['trn']['x'].pop(val_idx[ii])
                data[tt]['trn']['y'].pop(val_idx[ii])     

    for tt in range(total_task):
        data[tt]["classes"] = np.unique(data[tt]["trn"]["y"])
        data[tt]['ncla'] = len(np.unique(data[tt]['trn']['y']))

    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n    

    # for i in range(num_tasks):
    #     cur = data[i]
    #     print(f"classes: {cur['classes']}")
    # exit()

    return data, taskcla, class_order

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

def get_params():
    model_ = models.vit_b_16
    weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
    vit_transform = weights.transforms()
    return_nodes = ["getitem_5"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = Extractor(model_, weights=weights, return_nodes=return_nodes, device=device)
    return vit_transform, device, extractor

def extract_core50():    
    vit_transform, device, extractor = get_params()

    path = "core50/core50_128x128/"    
    outfile = "ds_core50_embedding.pt"

    # make static so they're in order
    # dirs = sorted(os.listdir(path))
    dirs = [f"s{i}" for i in range(1, 12)]
    embs = {}
    for dir in dirs:
        imagefolder_path = os.path.join(os.getcwd(), path, dir)
        print(imagefolder_path)
    
        dataset = datasets.ImageFolder(imagefolder_path, transform=vit_transform)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
        embeddings, targets = [], []
        for images, labels_ in dataloader:
            extracted = extractor(images.to(device))
            e = extracted.detach().cpu()
            l = labels_.detach().cpu().numpy().tolist()
            embeddings.extend(zip(e, l))
            targets.extend(l)
            print(np.unique(labels_))
            torch.cuda.empty_cache()
        embs[dir] = {"data": embeddings, "targets": targets}
    
    torch.save(embs, outfile)

def extract_inaturalist():
    vit_transform, device, extractor = get_params()

    path = "dataset/inaturalist/train_val2018"
    outfile = "ds_inaturalist_embedding.pt"    
    dirs = ["Actinopterygii", "Amphibia", "Animalia",
            "Arachnida", "Aves", "Bacteria", "Chromista",
            "Fungi", "Insecta", "Mammalia", "Mollusca",
            "Plantae", "Protozoa", "Reptilia"]
    embs = {}
    for dir in dirs:
        imagefolder_path = os.path.join(os.getcwd(), path, dir)
        print(imagefolder_path)

        dataset = datasets.ImageFolder(imagefolder_path, transform=vit_transform)
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)
        embeddings, targets = [], []
        for images, labels_ in dataloader:
            extracted = extractor(images.to(device))
            e = extracted.detach().cpu()
            l = labels_.detach().cpu().numpy().tolist()
            embeddings.extend(zip(e, l))
            targets.extend(l)
            print(np.unique(labels_))
            torch.cuda.empty_cache()
        embs[dir] = {"data": embeddings, "targets": targets, "classes": dataset.classes, "class_to_idx": dataset.class_to_idx}       
    torch.save(embs, outfile) 

def extract_oxflowers():
    vit_transform, device, extractor = get_params()
    path = "dataset/oxford_flowers/"
    outfile = "ds_oxford_flowers.pt"
    embs = {}
    imagefolder_path = os.path.join(os.getcwd(), path, "jpg")
    print(imagefolder_path)
    dataset = datasets.ImageFolder(imagefolder_path, transform=vit_transform)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)
    embeddings = []
    for images, _ in dataloader:
        extracted = extractor(images.to(device))
        e = extracted.detach().cpu()
        embeddings.extend(e)
        torch.cuda.empty_cache()
        print(images.size())

    # attach labels
    labels = scipy.io.loadmat(os.path.join(os.getcwd(), path, "imagelabels.mat"))
    labels = labels["labels"].tolist()[0]
    embeddings = [(feat, lab) for feat, lab in zip(embeddings, labels)]

    # attach setid
    setid = scipy.io.loadmat(os.path.join(os.getcwd(), path, "setid.mat"))
    trnid = setid["trnid"].tolist()[0]
    valid = setid["valid"].tolist()[0]
    tstid = setid["tstid"].tolist()[0]
    embs = {"data": embeddings, "targets": labels, "trnid": trnid, "valid": valid, "tstid": tstid}
    torch.save(embs, outfile)

def extract_mitscenes():
    vit_transform, device, extractor = get_params()
    path = "dataset/mit_scenes/"
    outfile = "mit_scenes.pt"

    imagefolder_path = os.path.join(os.getcwd(), path, "Images")
    dataset = datasets.ImageFolder(imagefolder_path, transform=vit_transform)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)
    embeddings, targets = [], []
    for images, labels_ in dataloader:
        extracted = extractor(images.to(device))
        e = extracted.detach().cpu()
        l = labels_.detach().cpu().numpy().tolist()
        print(np.unique(labels_))
        embeddings.extend(zip(e, l))
        targets.extend(l)
        torch.cuda.empty_cache()
    embs = {"data": embeddings, "targets": targets, "classes": dataset.classes, "class_to_idx": dataset.class_to_idx}
    torch.save(embs, outfile)

def split_mitscenes():
    mit = torch.load("mit_scenes.pt")
    path = "dataset/mit_scenes/"
    outfile = "mit_scenes.pt"
    train_all, test_all, train_names, test_names = [], [], {}, {}

    # READ TRAIN/TEST    
    with open(os.path.join(os.getcwd(), path, "TrainImages.txt")) as f:
        train_lines = f.readlines()

    with open(os.path.join(os.getcwd(), path, "TestImages.txt")) as f:
        test_lines = f.readlines()

    for line in train_lines:
        line = line.strip().split("/")
        existing_dir = train_names.get(line[0], [])
        existing_dir.append(line[1])
        train_names[line[0]] = existing_dir

    for line in test_lines:
        line = line.strip().split("/")
        existing_dir = test_names.get(line[0], [])
        existing_dir.append(line[1])
        test_names[line[0]] = existing_dir        

    for dir in mit["classes"]:
        images_path = os.path.join(os.getcwd(), path, "Images", dir)
        files = os.listdir((images_path))
        train_idx, test_idx = np.zeros(len(files)), np.zeros(len(files))

        for name in train_names[dir]:
            train_idx[files.index(name)] = 1
        for name in test_names[dir]:
            test_idx[files.index(name)] = 1

        train_all.extend(train_idx.tolist())
        test_all.extend(test_idx.tolist())
    
    train_all = (torch.tensor(train_all).int() == 1).nonzero().squeeze()
    test_all = (torch.tensor(test_all).int() == 1).nonzero().squeeze()

    print(train_all.size())
    print(test_all.size())

    features = [feat[0] for feat in mit["data"]]
    labels = [feat[1] for feat in mit["data"]]
    features = torch.stack(features)
    labels = torch.tensor(labels).int()
    print(features.size(), labels.size())

    train_features = torch.index_select(features, 0, train_all)
    test_features = torch.index_select(features, 0, test_all)
    train_labels = torch.index_select(labels, 0, train_all).cpu().tolist()
    test_labels = torch.index_select(labels, 0, test_all).cpu().tolist()

    train_features = [(f, l) for f, l in zip(train_features, train_labels)]
    test_features = [(f, l) for f, l in zip(test_features, test_labels)]

    train_ds = {"data": train_features, "targets": train_labels, "classes": mit["classes"], "class_to_idx": mit["class_to_idx"]}
    test_ds = {"data": test_features, "targets": test_labels, "classes": mit["classes"], "class_to_idx": mit["class_to_idx"]}

    torch.save(train_ds, "ds_mit_scenes_train.pt")
    torch.save(test_ds, "ds_mit_scenes_test.pt")

def separate_cub():
    path = os.path.join(os.getcwd(), "dataset/cub/CUB_200_2011")
    name_split = {}
    with open(os.path.join(path, "images.txt")) as f1, open(os.path.join(path, "train_test_split.txt")) as f2:
        images = f1.readlines()
        datasplit = f2.readlines()
        for image, split_ in zip(images, datasplit):
            _, names = image.strip().split(" ")
            _, name = names.split("/")          
            tsplit = int(split_.strip().split(" ")[1])
            name_split[name] = tsplit

    train_path = os.path.join(path, "train")
    test_path = os.path.join(path, "test")
    images_path = os.path.join(path, "images")
    dirs = sorted(os.listdir(images_path))
    for dir in dirs:        
        if not os.path.exists(os.path.join(train_path, dir)):
            os.mkdir(os.path.join(train_path, dir))
        if not os.path.exists(os.path.join(test_path, dir)):
            os.mkdir(os.path.join(test_path, dir))            
        files = os.listdir(os.path.join(images_path, dir))
        for file in files:
            split_ = name_split[file]
            current = os.path.join(images_path, dir, file)
            destination = os.path.join(train_path, dir, file) if split_ == 1 else os.path.join(test_path, dir, file)
            os.rename(current, destination)                        

def separate_stanfordcars():
    path = os.path.join(os.getcwd(), "dataset/stanford_cars")
    anno = scipy.io.loadmat(os.path.join(path, "cars_annos.mat"))

    name_split = {}
    for img in anno["annotations"][0]:
        name = img[0][0].split("/")[1]
        name_split[name] = (img[5][0][0], img[6][0][0])

    train_path = os.path.join(path, "train")
    test_path = os.path.join(path, "test")
    images_path = os.path.join(path, "car_ims")
    files = os.listdir(images_path)
    for file in files:
        cls_, test = name_split[file]
        if not os.path.exists(os.path.join(train_path, str(cls_))):
            os.mkdir(os.path.join(train_path, str(cls_)))
        if not os.path.exists(os.path.join(test_path, str(cls_))):
            os.mkdir(os.path.join(test_path, str(cls_)))            
        current = os.path.join(images_path, file)
        destination = os.path.join(test_path, str(cls_), file) if test == 1 else os.path.join(train_path, str(cls_), file)
        os.rename(current, destination)        

def separate_fgvcaircraft():
    path = os.path.join(os.getcwd(), "dataset/fgvc_aircraft/fgvc-aircraft-2013b/data")
    train_path = os.path.join(path, "images_variant_train.txt")
    val_path = os.path.join(path, "images_variant_val.txt")
    test_path = os.path.join(path, "images_variant_test.txt")

    with open(train_path) as tr, open(val_path) as va, open(test_path) as te:
        train_idx = [name.strip() for name in tr.readlines()]
        val_idx = [name.strip() for name in va.readlines()]
        test_idx = [name.strip() for name in te.readlines()]

    # need to replace dash -, space  and forward slash / to underscore _
    # use stupid/lazy way
    train_idx = {t.split(" ")[0]: t.split(" ")[1].replace("-", "_").replace(" ", "_").replace("/", "_") for t in train_idx}
    val_idx = {t.split(" ")[0]: t.split(" ")[1].replace("-", "_").replace(" ", "_").replace("/", "_") for t in val_idx}
    test_idx = {t.split(" ")[0]: t.split(" ")[1].replace("-", "_").replace(" ", "_").replace("/", "_") for t in test_idx}

    train_path = os.path.join(path, "train")
    val_path = os.path.join(path, "val")
    test_path = os.path.join(path, "test")
    images_path = os.path.join(path, "images")
    files = os.listdir(images_path)
    for file in files:
        name = file[:-4]
        if name in train_idx:
            cls_ = train_idx[name]
            if not os.path.exists(os.path.join(train_path, cls_)):
                os.mkdir(os.path.join(train_path, cls_))
            destination = os.path.join(train_path, cls_, file)
        elif name in val_idx:
            cls_ = val_idx[name]
            if not os.path.exists(os.path.join(val_path, cls_)):
                os.mkdir(os.path.join(val_path, cls_))
            destination = os.path.join(val_path, cls_, file)                
        elif name in test_idx:   
            cls_ = test_idx[name]
            if not os.path.exists(os.path.join(test_path, cls_)):
                os.mkdir(os.path.join(test_path, cls_))       
            destination = os.path.join(test_path, cls_, file)     
        current = os.path.join(images_path, file)
        print(current, destination)
        os.rename(current, destination)     

def separate_letters():
    path = os.path.join(os.getcwd(), "dataset/letters")
    train_path = os.path.join(path, "good_train.txt")
    val_path = os.path.join(path, "good_val.txt")
    test_path = os.path.join(path, "good_test.txt")

    with open(train_path) as tr, open(val_path) as va, open(test_path) as te:
        train_idx = [name.strip() for name in tr.readlines()]
        val_idx = [name.strip() for name in va.readlines()]
        test_idx = [name.strip() for name in te.readlines()]

    # English/Img/GoodImg/Bmp/Sample050/img050-00009.png
    # make a dictionary of {class: [images]} for lookup
    train_idx = letters_helper(train_idx, dict())
    val_idx = letters_helper(val_idx, dict())
    test_idx = letters_helper(test_idx, dict())

    train_path = os.path.join(path, "train")
    val_path = os.path.join(path, "val")
    test_path = os.path.join(path, "test")
    images_path = os.path.join(path, "English/Img/GoodImg/Bmp")
    dirs = os.listdir(images_path)

    for dir in dirs:
        if not os.path.exists(os.path.join(train_path, dir)):
            os.mkdir(os.path.join(train_path, dir))
        if not os.path.exists(os.path.join(val_path, dir)):
            os.mkdir(os.path.join(val_path, dir))
        if not os.path.exists(os.path.join(test_path, dir)):
            os.mkdir(os.path.join(test_path, dir))

        files = os.listdir(os.path.join(images_path, dir))
        for file in files:
            if file in train_idx[dir]:
                destination = os.path.join(train_path, dir, file)
            elif file in val_idx[dir]:
                destination = os.path.join(val_path, dir, file)
            elif file in test_idx[dir]:
                destination = os.path.join(test_path, dir, file)
            current = os.path.join(images_path, dir, file)
            print(current, destination)
            os.rename(current, destination)

def letters_helper(idx, dict_):
    for idx_ in idx:
        idx_ = idx_.split("/")
        cls_, name = idx_[-2], idx_[-1]
        temp = dict_.get(cls_, [])
        temp.append(name)
        dict_[cls_] = temp
    return dict_
            
def extract_generic(subdir, outfile):
    vit_transform, device, extractor = get_params()
    path = os.path.join(os.getcwd(), subdir)

    dataset = datasets.ImageFolder(path, transform=vit_transform)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)
    embeddings, targets = [], []
    for images, labels_ in dataloader:
        extracted = extractor(images.to(device))
        e = extracted.detach().cpu()
        l = labels_.detach().cpu().numpy().tolist()
        print(np.unique(labels_))
        embeddings.extend(zip(e, l))
        targets.extend(l)
        torch.cuda.empty_cache()
    embs = {"data": embeddings, "targets": targets, "classes": dataset.classes, "class_to_idx": dataset.class_to_idx}
    torch.save(embs, outfile)

def conv2image_svhn(path, mat_file, type):
    path = os.path.join(os.getcwd(), path)
    dir_path = os.path.join(path, type)
    
    mat = scipy.io.loadmat(os.path.join(path, mat_file))
    X = np.transpose(mat["X"], (3, 0, 1, 2))
    y = mat["y"].reshape(-1).tolist()

    for i in range(len(y)):
        if not os.path.exists(os.path.join(dir_path, str(y[i]))):
            os.mkdir(os.path.join(dir_path, str(y[i])))
        im = Image.fromarray(X[i])        
        im.save(os.path.join(dir_path, str(y[i]), f"{str(time.time_ns())}.png"))

if __name__ == "__main__":    
    ###############################################
    # EXTRACT SVHN
    # conv2image_svhn("dataset/svhn", "train_32x32.mat", "train")
    # conv2image_svhn("dataset/svhn", "test_32x32.mat", "test")
    # extract_generic("dataset/svhn/train", "ds_svhn_train.pt")
    # extract_generic("dataset/svhn/test", "ds_svhn_test.pt")
    exit()            
    ###############################################
    # EXTRACT LETTERS
    # separate_letters()    
    # extract_generic("dataset/letters/train", "ds_letters_train.pt")
    # extract_generic("dataset/letters/val", "ds_letters_val.pt")
    # extract_generic("dataset/letters/test", "ds_letters_test.pt")
    exit()            
    ###############################################
    # EXTRACT FGVC_AIRCRAFT
    # separate_fgvcaircraft()
    # extract_generic("dataset/fgvc_aircraft/fgvc-aircraft-2013b/data/train", "ds_fgvc_aircraft_train.pt")
    # extract_generic("dataset/fgvc_aircraft/fgvc-aircraft-2013b/data/val", "ds_fgvc_aircraft_val.pt")
    # extract_generic("dataset/fgvc_aircraft/fgvc-aircraft-2013b/data/test", "ds_fgvc_aircraft_test.pt")
    exit()        
    ###############################################
    # EXTRACT iNATURALIST
    extract_inaturalist()
    exit()    
    ###############################################
    # EXTRACT CUB
    # separate_stanfordcars()
    # extract_generic("dataset/stanford_cars/train", "ds_stanford_cars_train.pt")
    # extract_generic("dataset/stanford_cars/test", "ds_stanford_cars_test.pt")
    exit()         
    ###############################################
    # EXTRACT CUB
    # separate_cub()    
    # extract_generic("dataset/cub/CUB_200_2011/train", "ds_cub_train.pt")
    # extract_generic("dataset/cub/CUB_200_2011/test", "ds_cub_test.pt")
    exit()        
    ###############################################
    # EXTRACT MIT SCENES
    # extract_mitscenes()
    # split_mitscenes()
    exit()    
    ###############################################
    # EXTRACT OXFORD FLOWERS
    extract_oxflowers()
    exit()
    ###############################################
    # EXTRACT CORE50
    extract_core50()
    exit()
    ###############################################

    train_embedding_path = "cifar100_train_embedding.pt"
    test_embedding_path = "cifar100_test_embedding.pt"
    val_embedding_path = None
    n_experts = 5

    data, class_order = get_data(
        train_embedding_path, 
        None, 
        val_embedding_path=test_embedding_path,
        num_tasks=n_experts, # num_tasks == n_experts
        expert=True
    )

    print(class_order)
    print(data.keys())
    print(data[0].keys())
    print(len(data[0]["train"]["x"]))
    print(len(data[0]["train"]["y"]))
    print(len(data[0]["val"]["x"]))
    print(len(data[0]["val"]["x"]))
    print(len(data[0]["test"]["x"]))
    print(len(data[0]["test"]["x"]))
    print(data[0]["classes"])
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


