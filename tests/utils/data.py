import os.path as osp
import torch
from torch_geometric.data import Data
from torch_geometric.utils import sort_edge_index
import os

def save_data(raw_dir, graph_data, file_name, graph_idx = None):
    r"""
    """
    if graph_idx is not None:
        torch.save(graph_data, osp.join(raw_dir,file_name+f'_{graph_idx}.pt'))
    else:
        torch.save(graph_data, osp.join(raw_dir,'/'+file_name+'.pt'))

def save_initial_model(path, name, model):
    r"""
    """
    save_dir = osp.join(path, "initialModels")
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)

    save_prefix = osp.join(save_dir, name+".pkl")
    #save_path = '{}_epoch_{}'.format(save_prefix, epoch)

    # logger.info("saving initial model to %s",save_prefix)
    output = open(save_prefix, mode="wb")
    torch.save({
            'model_state_dict': model.state_dict(),
            }, output)
    output.close()

def shuffle_graph(x, edge_index, perm):
    r"""
    """
    nums = torch.bincount(edge_index[0])
    row1 = perm.repeat_interleave(nums)
    ends = []
    for node in range(edge_index[0][-1] + 1):
        ends.extend([perm[edge_index[1][i]].item() for i in torch.where(edge_index[0] == node)[0]])
    return x[perm,:], sort_edge_index(torch.stack((row1, torch.tensor(ends)), dim=0))

class PairData(Data):
    r"""
    """
    def __init__(self, edge_index_s=None, x_s=None, 
                edge_index_t=None, x_t=None, y=None):
        super(PairData, self).__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.edge_index_t = edge_index_t
        self.x_t = x_t
        self.y = y

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __repr__(self):
        return '{}(x_s = {}, edge_index_s = {}, x_t = {}, edge_index_t = {}, y = {})'.format(
            self.__class__.__name__, self.x_s.shape, self.edge_index_s.shape,
            self.x_t.shape, self.edge_index_t.shape, self.y.shape
        )
    
class EarlyStoppingModule(object):
    """
    Module to keep track of validation score across epochs
    Stop training if score not improving exceeds patience
    """  
    def __init__(self, patience=100, delta=0.0001):
        self.patience = patience 
        self.delta = delta
        self.best_scores = None
        self.num_bad_epochs = 0 
        self.should_stop_now = False

    def save_best_model(self, path, model, name, epoch): 
        save_dir = os.path.join(path, "bestValidationModels")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        #name = name + "_tfrac_" + str(self.av.TEST_FRAC) + "_vfrac_" + str(self.av.VAL_FRAC)
        save_path = os.path.join(save_dir, name)

        output = open(save_path, mode="wb")
        torch.save({
                'model_state_dict': model.state_dict(),
                'epoch':epoch,
                }, output)
        output.close()

    def load_best_model(self, path, name):
        load_dir = os.path.join(path, "bestValidationModels")
        if not os.path.isdir(load_dir):
            raise Exception('{} does not exist'.format(load_dir))
        #name = name + "_tfrac_" + str(self.av.TEST_FRAC) + "_vfrac_" + str(self.av.VAL_FRAC)
        load_path = os.path.join(load_dir, name)
        checkpoint = torch.load(load_path)
        return checkpoint

    def diff(self, curr_scores):
        return sum ([cs-bs for cs,bs in zip(curr_scores, self.best_scores)])

    def check(self,curr_scores,model,epoch) :
        if self.best_scores==None: 
            self.best_scores = curr_scores
            self.save_best_model(model,epoch)
        elif self.diff(curr_scores) >= self.delta:
            self.num_bad_epochs = 0
            self.best_scores = curr_scores
            self.save_best_model(model,epoch)
        else:  
            self.num_bad_epochs+=1
        if self.num_bad_epochs>self.patience: 
            self.should_stop_now = True
        return self.should_stop_now  