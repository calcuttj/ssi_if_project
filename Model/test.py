from argparse import ArgumentParser as ap
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('TkAgg')
from iftool.gnn_challenge import ShowerFeatures
from torch_geometric.loader import DataLoader as GraphDataLoader
from torch_geometric.data import Data as GraphData

import numpy as np
from torch_geometric.nn import EdgeConv, global_mean_pool
import h5py as h5
import time, calendar
from EdgeConvNet import EdgeConvNet


def make_net(args):
  net = EdgeConvNet()

  net.load_state_dict(
    torch.load(args.pt)['model_state_dict']
  )
  return net

if __name__ == '__main__':
  parser = ap()
  parser.add_argument('-i', required=True, type=str)
  parser.add_argument('--pt', required=True, type=str)
  parser.add_argument('--max', type=int, default=-1)
  args = parser.parse_args()


  test_data = ShowerFeatures(file_path=args.i)
  loader = GraphDataLoader(test_data,
                                shuffle=False,
                                num_workers = 4,
                                batch_size  = 64,
                                pin_memory=False,
                               )
  net = make_net(args)
  net.eval()

  loss_fn = torch.nn.BCELoss(reduction='mean')

  node_event_v, edge_event_v     = [], [] # Stores the event ID of each node/edge
  node_pred_v, edge_pred_v       = [], [] # Stores the binary prediction for node and edges (argmax)
  node_sigmoid_v, edge_sigmoid_v = [], [] # Stores the sigmoid score for the primary and ON channel of node and edges, respectively
  node_label_v, edge_label_v     = [], [] # Stores the true binary label for node and edges
  edge_index_v = [] # Stores the edge index for each event


  numbatches = 0
  with torch.set_grad_enabled(False):
    for data in loader:
      if args.max > 0 and numbatches >= args.max: break
      numbatches += 1
      # Bring data to GPU, if requested
      ##if device != 'cpu':
      ##    data = data.to(torch.device(device))

      node_pred, edge_pred = net(data)

      if node_pred is not None:
        node_pred_v.append( np.round(node_pred).flatten())
        node_sigmoid_v.append(node_pred.flatten())
        node_label_v.append( data.y )

      if edge_pred is not None:
        edge_pred_v.append( np.round(edge_pred).flatten())
        edge_sigmoid_v.append(edge_pred.flatten())
        edge_label_v.append( data.edge_label )

      #cids = np.concatenate([np.arange(c) for c in np.unique(data.batch.numpy(),return_counts=True)[1]])
      #edge_index_v.append( cids[data.edge_index.numpy()].T )

      #print(node_pred_v[-1], node_label_v[-1])

  if len(node_label_v) > 0:
    node_label = np.concatenate(node_label_v)
    node_pred = np.concatenate(node_pred_v)
    node_sigmoid = np.concatenate(node_sigmoid_v)
    secondary_mask, primary_mask = node_label==0, node_label==1
    print('Secondary classification accuracy:', np.sum(node_label[secondary_mask]==node_pred[secondary_mask])/np.sum(secondary_mask))
    print('Primary classification accuracy:', np.sum(node_label[primary_mask]==node_pred[primary_mask])/np.sum(primary_mask))

    plt.figure(figsize=(12,8))
    plt.hist(node_sigmoid[node_label==1], bins=20, range=[0,1.0001], label='True primaries', alpha=0.5)
    plt.hist(node_sigmoid[node_label==0], bins=20, range=[0,1.0001], label='True secondaries', alpha=0.5)
    plt.yscale('log')
    plt.legend()
    plt.draw()
    plt.show()

  if len(edge_label_v) > 0:
    edge_label = np.concatenate(edge_label_v)
    edge_pred = np.concatenate(edge_pred_v)
    edge_sigmoid = np.concatenate(edge_sigmoid_v)



    # Draw the sigmoid score of edges
    from matplotlib import pyplot as plt
    plt.figure(figsize=(12,8))
    plt.hist(edge_sigmoid[edge_label==1], bins=20, range=[0,1.0001], label='True ON edges', alpha=0.5)
    plt.hist(edge_sigmoid[edge_label==0], bins=20, range=[0,1.0001], label='True OFF edges', alpha=0.5)
    #plt.yscale('log')
    plt.legend()
    plt.draw()
    plt.show()
    
    off_mask, on_mask = edge_label==0, edge_label==1
    print('OFF edge classification accuracy:', np.sum(edge_label[off_mask]==edge_pred[off_mask])/np.sum(off_mask))
    print('ON edge classification accuracy:', np.sum(edge_label[on_mask]==edge_pred[on_mask])/np.sum(on_mask))
