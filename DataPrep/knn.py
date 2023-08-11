import torch

# Install packages
!pip uninstall torch-scatter torch-sparse torch-geometric torch-cluster  --y
!pip install torch-scatter -f https://data.pyg.org/whl/torch-{torch.__version__}.html
!pip install torch-sparse -f https://data.pyg.org/whl/torch-{torch.__version__}.html
!pip install torch-cluster -f https://data.pyg.org/whl/torch-{torch.__version__}.html
!pip install git+https://github.com/pyg-team/pytorch_geometric.git

# Import knn
from torch_cluster import knn_graph

# Import as a PyTorch geometric objects with the following attributes:
#
# x: A (C,16) tensor of node features
#	[0-2: pos, 3-11: cov, 12-14: ???, 15: voxel count]
# edge_index: a (2,E) sparse representation of the adjacency matrix
#	[1, 2, ..., C-1, 0, 2, ...]
#	[0, 0, ...,  0 , 1, 1, ...]
# edge_attr: a (E,19) tensor of edge features
#	[0-2: frag_1 pos, 3-5: frag_2 pos, 6-8: unit dirVector, 9: distance, 10-18: outer product]	
# y: a (C) vector of node labels (primary IDs)
# edge_label: a (E) vector of edge labels (1 if connects two nodes in the same group, 0 otherwise)
# index: a scalar representing the entry indes
#
from iftool.gnn_challenge import ShowerFeatures
data = ShowerFeatures(file_path = datapath)

# I wanted to get knn with distance defined as (min voxel distance b/w nodes)
# But knn_graph interface does not support self-defined distance function
# We can use this for now
def getKNNEdge(data, k)
	return knn_graph(data.x[:, :3], k=k, batch=data.batch) 

# Given an edge: edge[1] --> edge[0]
#   and number of nodes (nNodes)
# Returns: edge index used for edge feature
def findEdgeIndex(edge, nNodes):
  index = edge[1] * (nNodes - 1)
  if edge[0] < edge[1]:
    index = index + edge[0] + 1
  else:
    index = index + edge[0]
  return index.item() - 1

# example:

x = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
batch = torch.tensor([0, 0, 0, 0])
edge_index = knn_graph(x[:,0], k=1, batch=batch)
print(edge_index)

for i in range(edge_index.size(dim=1)):
  print(edge_index[1, i].item(), " to ", edge_index[0, i].item(), ": ", findEdgeIndex(edge_index[:, i], 4))
###


