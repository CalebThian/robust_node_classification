# Reference: https://blog.csdn.net/weixin_41650348/article/details/112754933

from torch_geometric.datasets import Planetoid
import torch

def download_data(dataset_name):
    if dataset_name == 'Cora':
        dataset = Planetoid(root='./cora/',name = 'Cora')
    elif dataset_name == 'Citeseer':
        dataset = Planetoid(root='./citeseer/',name = 'Citeseer')
    elif dataset_name == 'Pubmed':
        dataset = Planetoid(root = './pubmed/', name = 'Pubmed')
    else:
        print('Dataset name not exists, only for Cora, Citaseer, Pubmed')
        dataset = None
    return dataset

def download_all_data():
    dataset_name = ['Cora','Citeseer','Pubmed']
    datasets = []
    for dset in  dataset_name:
        datasets.append(download_data(dset))
    return datasets