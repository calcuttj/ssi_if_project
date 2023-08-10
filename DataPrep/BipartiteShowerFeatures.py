import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data as GraphData
from torch_geometric.data import Dataset as GraphDataset

class BipartiteShowerFeatures(ShowerFeatures):

  def __getitem__(self, idx):
        
    # Get the subset of node and edge features that correspond to the requested event ID
    if self._file_handle is None:
        import h5py
        self._file_handle = h5py.File(self._file_path, "r", swmr=True)
        
    node_info = torch.tensor(self._file_handle['node_features'][idx].reshape(-1, 19), dtype=torch.float32)
    node_features, group_ids, node_labels = node_info[:,:-3], node_info[:,-2].long(), node_info[:,-1].long()
    
    edge_info = torch.tensor(self._file_handle['edge_features'][idx].reshape(-1, 22), dtype=torch.float32)
    edge_features, edge_index, edge_labels = edge_info[:,:-3], edge_info[:,-3:-1].long().t(), edge_info[:,-1].long()

    indices = [node_labels[y[0]] != node_labels[y[1]] for y in edge_index.T]

    print(edge_index.T)
    print(np.where(indices))
    return GraphData(x = node_features,
                    edge_index = edge_index[:, indices],
                    edge_attr = edge_features[indices],
                    y = node_labels,
                    edge_label = edge_labels[indices],
                    index = idx)
