from graph_match.sgmatch.models.SimGNN import SimGNN
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import OneHotDegree
import torch
import os.path as osp
import random
import numpy as np

random.seed(0)

DATASET_NAME = 'MUTAG'
BATCH_SIZE = 32
DATASET_SIZE = 188

path = osp.join('data', DATASET_NAME)
dataset = TUDataset(path, name = DATASET_NAME, use_node_attr=True, use_edge_attr=True).shuffle()

dataset = dataset.shuffle()
train_graphs = dataset[:np.round(0.6*DATASET_SIZE)]
val_graphs = dataset[np.round(0.6*DATASET_SIZE):np.round(0.75*DATASET_SIZE)]
test_graphs = dataset[141:188]