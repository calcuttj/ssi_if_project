import torch.nn as nn
from torch_geometric.nn import EdgeConv, global_mean_pool
import torch.nn.functional as F

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
  def __init__(self, n_node_features, aggr='mean'):
    super().__init__()

    assert_msg = 'aggr should be one of mean, max, or min'
    assert (aggr in ['mean', 'max', 'min']), assert_msg



    self.relu1 = nn.ReLU()
    self.edge_conv = EdgeConv(InnerNet(n_node_features, 64), aggr=aggr)
    layers = [
        nn.Linear(64 + n_node_features, 128),
        nn.ReLU(),
        nn.Linear(128, 1),

    ]
    self.end_layers = nn.Sequential(*layers)

  def forward(self, data):
    x = torch.cat((self.edge_conv(data.x, data.edge_index), data.x),
                  dim=1)
  
    self.relu1(x)
    x = global_mean_pool(x, data.batch)
    return F.sigmoid(self.end_layers(x))



