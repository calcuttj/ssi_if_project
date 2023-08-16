from torch_geometric.nn.glob import global_mean_pool
from argparse import ArgumentParser as ap
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool import global_mean_pool
import matplotlib.pyplot as plt
from iftool.gnn_challenge import ShowerFeatures
train_data = ShowerFeatures(file_path = datapath)
from torch_geometric.loader import DataLoader as GraphDataLoader
from torch_geometric.data import Data as GraphData
train_loader = GraphDataLoader(train_data,
                               shuffle     = True,
                               num_workers = 4,
                               batch_size  = 64
                              )

import h5py as h5
import numpy as np
datapath='if-graph-train.h5'

# Open a file in 'r'ead mode.
f=h5.File(datapath,mode='r',swmr=True)

# List items in the file
for key in f.keys():
    print('dataset',key,'... type',f[key].dtype,'... shape',f[key].shape)


import torch.nn as nn
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


import h5py as h5
import time, calendar

def train(net, optimizer, loss_fn, nepochs=1, save=False, maxbatches=-1):
  # check if a GPU is available. Otherwise run on CPU
  device = 'cpu'
  args_cuda = torch.cuda.is_available()
  if args_cuda: device = "cuda:0"
  print('device : ',device)
  net.to(device)

  train_file = f'training_stats_{calendar.timegm(time.gmtime())}.h5'
  fout = h5.File(train_file, 'a')

  epoch_losses = []
  epoch_edge_losses = []
  epoch_node_losses = []
  nepochs = 1
  net.train()
  for i in range(nepochs):
    print(f'EPOCH {i}')
    running_loss = 0.
    losses = []
    edge_losses = []
    node_losses = []
    for batchnum, batch in enumerate(train_loader):
      if maxbatches > 0 and batchnum >=maxbatches: break
      optimizer.zero_grad()
      batch.to(device)
      pred_n, pred_e = net(batch)
      loss_e = loss_fn(pred_e, batch.edge_label.reshape(len(batch.edge_label), 1).float())
      loss_n = loss_fn(pred_n, batch.y.reshape(len(batch.y), 1).float())

      loss = loss_n + loss_e
      loss.backward()
      optimizer.step()
      theloss = loss.item()
      running_loss += theloss
      if not batchnum % 10: print(f'{batchnum}')
      if not batchnum % 100 and batchnum > 0:
        print(f'\n(Batch {batchnum}) Loss: {running_loss / 100.}')
        running_loss = 0.
      losses.append(theloss)
      edge_losses.append(loss_e.item())
      node_losses.append(loss_n.item())
    epoch_losses.append(losses)
    epoch_node_losses.append(node_losses)
    epoch_edge_losses.append(edge_losses)

    if save:
      state = {
        'epoch':i,
        'model_state_dict':net.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
      }
      cpt_file = f'checkpoint_{calendar.timegm(time.gmtime())}.pt'
      torch.save(
        state,
        cpt_file
      )

      print('Saved checkpoint file:', cpt_file)

  fout.create_dataset('losses', data=np.array(epoch_losses))
  fout.create_dataset('edge_losses', data=np.array(epoch_edge_losses))
  fout.create_dataset('node_losses', data=np.array(epoch_node_losses))

  print('Saved training file:', train_file)
  fout.close()


def make_trainers():
  net = EdgeConvNet()
  loss_fn = torch.nn.BCELoss(reduction='mean')
  optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
  return net, optimizer, loss_fn

if __name__ = '__main__':
  
  parser = ap()
  parser.add_argument('--save', action='store_true')
  args = parser.parse_args()

  net, optimizer, loss_fn = make_trainers()

  train(net, optimizer, loss_fn, save=args.save)
