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
