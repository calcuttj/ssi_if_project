import torch.nn as nn
import torch
from torch_geometric.nn import EdgeConv, global_mean_pool
import torch.nn.functional as F

class NodeMixer(torch.nn.Module):
  def __init__(self, the_op=torch.sub):
    super().__init__()
    self.the_op = the_op
  def forward(self, x, edge_index):
    return self.the_op(x[edge_index[0]], x[edge_index[1]])


class InnerNet(torch.nn.Module):
  def __init__(self, n_node_features, layer_width):
    super().__init__()

    layers = []

    for i in range(3):
      insize = (2*n_node_features if i == 0 else layer_width)
      layers.append(nn.Linear(insize, layer_width))
      layers.append(nn.BatchNorm1d(num_features=layer_width))
      layers.append(nn.ReLU())

    self.model = nn.Sequential(*layers)


  def forward(self, data):
    return self.model(data)


class EdgeConvNet(torch.nn.Module):
  def __init__(self, n_node_features=16, aggr='mean'):
    super().__init__()

    assert_msg = 'aggr should be one of mean, max, or min'
    assert (aggr in ['mean', 'max', 'min']), assert_msg



    #self.relu1 = nn.ReLU()
    self.edge_conv = EdgeConv(InnerNet(n_node_features, 64), aggr=aggr)
    self.edge_conv2 = EdgeConv(InnerNet(64 + n_node_features, 128), aggr=aggr)

    node_layers = [
        nn.Linear(128 + 64 + n_node_features, 256),
        nn.BatchNorm1d(num_features=256),
        nn.ReLU(),
        nn.Linear(256, 1),
    ]
    self.end_node_layers = nn.Sequential(*node_layers)

    #self.node_mixer_op = InnerNet(128+64+n_node_features, 128+64+n_node_features)
    self.node_mixer = NodeMixer()
    edge_layers = [
        nn.Linear(128 + 64 + n_node_features, 256),
        nn.BatchNorm1d(num_features=256),
        nn.ReLU(),
        nn.Linear(256, 1),
    ]
    self.end_edge_layers = nn.Sequential(*edge_layers)    


  def forward(self, data):
    x = torch.cat((self.edge_conv(data.x, data.edge_index), data.x),
                  dim=1) #output is shape [n_nodes, n_node_features + 64]
    x = torch.cat((self.edge_conv2(x, data.edge_index), x),
                  dim=1) #output is shape [n_nodes, n_node_feats + 64 + 128]

    #Branch 1: a MLP outputing in node space
    node_out = F.sigmoid(self.end_node_layers(x))

    #Branch 2: a MLP outputing in edge space
    #first, mix into edge space
    edge_out = self.node_mixer(x, data.edge_index)
    edge_out = F.sigmoid(self.end_edge_layers(edge_out))
    return (node_out, edge_out)



