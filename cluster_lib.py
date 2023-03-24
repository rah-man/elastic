import copy
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import warnings
import statistics

from collections import Counter
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from torch.utils.data import DataLoader

from base import BaseDataset, get_data, get_cifar100_coarse
from cluster import cluster_diff_alg, clusterKMEANS
from earlystopping import EarlyStopping
from sklearn.metrics import accuracy_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score

def gmm(x):
    """
    Run GMM for the given x.
    The number of components are 2 and 3.
    Return:
        - the model that gives the highest silhouette score
        - the predictions for x
    """
    n_components = np.arange(2, 4)
    models = []
    silhouette, predictions = [], []
    for n in n_components:
        m = GaussianMixture(n, covariance_type="full")
        pred = m.fit_predict(x)
        models.append(m)
        predictions.append(pred)       
        silhouette.append(silhouette_score(x, predictions[-1]))
    
    num_cluster = silhouette.index(max(silhouette)) + 2
    model = models[num_cluster-2]
    return model, predictions[num_cluster-2]

def class_gmm(x, y):
    """
    Run GMM for each class in x
    The number of components are 2 and 3.
    Return:
        - a dictionary of {class: gmm}
    """    
    class_gmm = {}
    y_unique = np.unique(y)
    for y_u in y_unique:
        x_ = x[y == y_u]
        model, _ = gmm(x_)
        class_gmm[y_u] = model
    return class_gmm        

def get_class_mean(x, y):
    """
    Calculate the class mean for given x and class y
    Return:
        - the class means
        - the order of y
    """    
    cls_means, cls_order = [], []
    y_unique = np.unique(y)
    for y_u in y_unique:
        x_ = x[y == y_u]
        cls_order.append(y_u)
        m_ = np.mean(x_, axis=0)
        cls_means.append(m_)
    return cls_means, cls_order

def class_mean_cluster(x, y, cls2cluster):
    """
    Run class mean clustering for the given x and y
    Return:
        - the model that gives the highest silhouette score
        - the mapping of class/label to cluster
        - the gmm for each class
    """    
    cls2cluster_ = {}
    cluster_shift = max(cls2cluster, key=int) + 1 if cls2cluster else 0
    cls_means, cls_order = get_class_mean(x, y)
    model, predictions = gmm(cls_means)

    cls2cluster_ = {y_: (p + cluster_shift) for y_, p in zip(cls_order, predictions)}
    # cls2gmm = class_gmm(x, y)
    # return model, cls2cluster_, cls2gmm
    return model, cls2cluster_
