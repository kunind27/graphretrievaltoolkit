import os.path as osp
import os
import random
from itertools import chain
from typing import Optional

import numpy as np
from torch_geometric.data import Dataset, Data
from torch_geometric.transforms import BaseTransform, Compose, NormalizeFeatures
import torch
import pickle

from .data import PairData

class Face2Edge(BaseTransform):
    def __call__(self, data):
        list_of_edges = []
        tri = data.face.t()
        
        def undirected_edge(a, b):
            return [[a,b], [b,a]]

        for triangle in tri:
            for e1, e2 in [[0,1],[1,2],[2,0]]: # for all edges of triangle
                list_of_edges.extend(undirected_edge(triangle[e1],triangle[e2])) # always lesser index first

        edge_index = np.unique(list_of_edges, axis=0).T # remove duplicates
        data.edge_index = torch.from_numpy(edge_index).to(device=data.face.device, dtype=torch.long)
        del data.face

        return data

def load_dataset(dpath: str, name: str, category: Optional[str] = None, train: Optional[bool] = None):
    d = None
    if not osp.isdir(dpath):
        os.makedirs(dpath)
    # fname = osp.join(dpath, name + ".pkl")
    # if os.path.isfile(fname):
    #     d = pickle.load(open(fname,"rb"))
    else:
        if name == "pascal_voc":
            from torch_geometric.datasets import PascalVOCKeypoints
            from torch_geometric.transforms import Delaunay

            transform = Compose([Delaunay(), Face2Edge()])
            assert category is not None, "Need to specify category for PascalVOCKeypoints"
            assert train is not None and isinstance(train, bool), "train parameter needs to be True or False"
            
            d = PascalVOCKeypoints(root=dpath, category=category, train=train, transform=transform)
        elif name == "citation_full":
            from torch_geometric.datasets import CitationFull
            assert category in ["Cora", "Cora_ML", "CiteSeer", "DBLP", "PubMed"], "Category should be one of ['Cora', 'Cora_ML' 'CiteSeer', 'DBLP', 'PubMed']"
            
            d = CitationFull(root=dpath, name=category)
        elif name == "TUDataset":
            from torch_geometric.datasets import TUDataset

            d = TUDataset(root=dpath, name='ENZYMES')
        elif name == "Planetoid":
            from torch_geometric.datasets import Planetoid
            assert category in ["Cora", "CiteSeer", "PubMed"], "Category should be one of ['Cora', 'CiteSeer', 'PubMed']"

            d = Planetoid(root=dpath, name=category)
        elif name == "GED":
            from torch_geometric.datasets import GEDDataset
            assert train is not None and isinstance(train, bool), "train parameter needs to be True or False"
            assert category in ["AIDS700nef", "LINUX", "ALKANE", "IMDBMulti"]
            d = GEDDataset(root=dpath, name=category, train=train)
        # with open(fname,"wb") as f:
        #     pickle.dump(d,f)
    assert d is not None
    
    return d

class SyntheticDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        with os.scandir(self.raw_dir) as files:
            return [file for file in files]

    @property
    def processed_file_names(self):
        with os.scandir(self.processed_dir) as files:
            return [file for file in files]
    
    def len(self):
        return len(self.processed_file_names) - 2

    def process(self):
        '''
        Reads graph data tuples saved in 'root/raw' directory 
        and processes and saves them as Data objects in
        'root/processed' directory.
        '''
        idx = 0
        paths = [x.split('.')[-2] for x in self.raw_paths]
        path_idxs = np.argsort(np.array([int(x.split('_')[-1]) for x in paths]))
        paths = [self.raw_paths[idx] for idx in path_idxs]

        for raw_path in paths:
            # Read data from `raw_path`.
            raw_data = torch.load(raw_path)
            if len(raw_data) == 3:
                data = Data(x = raw_data[0], edge_index = raw_data[1], y = raw_data[2])
            else:
                data = Data(x = raw_data[0], edge_index = raw_data[1])

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f'graph_{idx}.pt'))
            idx += 1

    def get(self, idx):
        '''
        Returns graph with index `idx` from self.root_dir
        '''
        graph = torch.load(osp.join(self.processed_dir, f'graph_{idx}.pt'))
        return graph

class PairDataset(torch.utils.data.Dataset):
    r"""Combines two datasets, a source dataset and a target dataset, by
    building pairs between separate dataset examples.
    Args:
        dataset_s (torch.utils.data.Dataset): The source dataset.
        dataset_t (torch.utils.data.Dataset): The target dataset.
        sample (bool, optional): If set to :obj:`True`, will sample exactly
            one target example for every source example instead of holding the
            product of all source and target examples. (default: :obj:`False`)
    """
    def __init__(self, dataset_s, dataset_t, sample=False, is_ged=False):
        self.dataset_s = dataset_s
        self.dataset_t = dataset_t
        self.sample = sample
        self.is_ged = is_ged

    def __len__(self):
        return len(self.dataset_s)

    def __getitem__(self, idx):
        if self.sample:
            data_s = self.dataset_s[idx]
            data_t = self.dataset_t[np.random.randint(0, len(self.dataset_t) - 1)]
        else:
            data_s = self.dataset_s[idx]
            data_t = self.dataset_t[idx]
        
        y = self.dataset_s.ged[data_s.i, data_t.i] if self.is_ged else data_t.y.t()

        return PairData(
            x_s=data_s.x,
            edge_index_s=data_s.edge_index,
            x_t=data_t.x,
            edge_index_t=data_t.edge_index,
            y=y
        )

    def __repr__(self):
        return '{}({}, {}, sample={})'.format(self.__class__.__name__,
                                              self.dataset_s, self.dataset_t,
                                              self.sample)

class ValidPairDataset(torch.utils.data.Dataset):
    r"""Combines two datasets, a source dataset and a target dataset, by
    building valid pairs between separate dataset examples.
    A pair is valid if each node class in the source graph also exists in the
    target graph.
    Args:
        dataset_s (torch.utils.data.Dataset): The source dataset.
        dataset_t (torch.utils.data.Dataset): The target dataset.
        sample (bool, optional): If set to :obj:`True`, will sample exactly
            one target example for every source example instead of holding the
            product of all source and target examples. (default: :obj:`False`)
    """
    def __init__(self, dataset_s, dataset_t, sample=False):
        self.dataset_s = dataset_s
        self.dataset_t = dataset_t
        self.sample = sample
        self.pairs, self.cumdeg = self.__compute_pairs__()

    def __compute_pairs__(self):
        num_classes = 0
        for data in chain(self.dataset_s, self.dataset_t):
            num_classes = max(num_classes, data.y.max().item() + 1)

        y_s = torch.zeros((len(self.dataset_s), num_classes), dtype=torch.bool)
        y_t = torch.zeros((len(self.dataset_t), num_classes), dtype=torch.bool)

        for i, data in enumerate(self.dataset_s):
            y_s[i, data.y] = 1
        for i, data in enumerate(self.dataset_t):
            y_t[i, data.y] = 1

        y_s = y_s.view(len(self.dataset_s), 1, num_classes)
        y_t = y_t.view(1, len(self.dataset_t), num_classes)

        pairs = ((y_s * y_t).sum(dim=-1) == y_s.sum(dim=-1)).nonzero()
        cumdeg = pairs[:, 0].bincount().cumsum(dim=0)

        return pairs.tolist(), [0] + cumdeg.tolist()

    def __len__(self):
        return len(self.dataset_s) if self.sample else len(self.pairs)

    def __getitem__(self, idx):
        if self.sample:
            data_s = self.dataset_s[idx]
            i = random.randint(self.cumdeg[idx], self.cumdeg[idx + 1] - 1)
            data_t = self.dataset_t[self.pairs[i][1]]
        else:
            data_s = self.dataset_s[self.pairs[idx][0]]
            data_t = self.dataset_t[self.pairs[idx][1]]

        y = data_s.y.new_full((data_t.y.max().item() + 1, ), -1)
        y[data_t.y] = torch.arange(len(data_t.y))
        y = y[data_s.y]
        y = torch.stack((torch.arange(len(data_s.y)), y), dim=0) # Mapping from idx of Gs to idx of Gt

        return PairData(
            x_s=data_s.x,
            edge_index_s=data_s.edge_index,
            x_t=data_t.x,
            edge_index_t=data_t.edge_index,
            y=y.transpose(0,-1)         
        )

    def __repr__(self):
        return '{}({}, {}, sample={})'.format(self.__class__.__name__,
                                              self.dataset_s, self.dataset_t,
                                              self.sample)