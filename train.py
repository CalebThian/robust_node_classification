# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 10:42:47 2023

@author: caleb
"""

import os
import argparse
import torch
import numpy as np
import scipy.sparse as sp
from baseline.GCN import GCN
from baseline.RSGNN import RSGNN
from dataset import Dataset, get_PtbAdj


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=False, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=12, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--estimator', type=str, default='MLP',
                    choices=['MLP','GCN'])
parser.add_argument('--mlp_hidden', type=int, default=64,
                    help='Number of hidden units of MLP graph constructor')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default="cora",
                    choices=['cora', 'cora_ml', 'citeseer','pubmed'], help='dataset')
parser.add_argument('--attack', type=str, default='random',
                    choices=['meta', 'random', 'nettack'])
parser.add_argument("--label_rate", type=float, default=0.01, 
                    help='rate of labeled data')
parser.add_argument('--ptb_rate', type=float, default=0.15, 
                    help="noise ptb_rate")
parser.add_argument('--epochs', type=int,  default=1000, 
                    help='Number of epochs to train.')

parser.add_argument('--alpha', type=float, default=0.01, 
                    help='weight of rec loss')
parser.add_argument('--sigma', type=float, default=100, 
                    help='the parameter to control the variance of sample weights in rec loss')
parser.add_argument('--beta', type=float, default=0.3, help='weight of label smoothness loss')
parser.add_argument('--threshold', type=float, default=0.8, 
                    help='threshold for adj of label smoothing')
parser.add_argument('--t_small',type=float, default=0.1,
                    help='threshold of eliminating the edges')

parser.add_argument('--inner_steps', type=int, default=2, 
                    help='steps for inner optimization')
parser.add_argument('--outer_steps', type=int, default=1, 
                    help='steps for outer optimization')
parser.add_argument('--lr_adj', type=float, default=0.001, 
                    help='lr for training adj')
parser.add_argument("--n_p", type=int, default=100, 
                    help='number of positive pairs per node')
parser.add_argument("--n_n", type=int, default=50, 
                    help='number of negitive pairs per node')

parser.add_argument("--r_type",type=str,default="flip",
                    choices=['add','remove','flip'])
args = parser.parse_known_args()[0]