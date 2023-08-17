from argparse import ArgumentParser as ap
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from iftool.gnn_challenge import ShowerFeatures
from torch_geometric.loader import DataLoader as GraphDataLoader
from torch_geometric.data import Data as GraphData

import numpy as np
from torch_geometric.nn import EdgeConv, global_mean_pool
import h5py as h5
import time, calendar
from EdgeConvNet import EdgeConvNet


def train(net, optimizer, loss_fn, train_loader, test_loader=None,
          nepochs=1, save=False, maxbatches=-1, do_test=False):
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

  epoch_test_losses = []
  epoch_test_edge_losses = []
  epoch_test_node_losses = []
  for i in range(nepochs):
    net.train()
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
      if not batchnum % 10: print(f'{batchnum}: Loss: {theloss}')
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
      cpt_file = f'checkpoint_epoch_{i}_{calendar.timegm(time.gmtime())}.pt'
      torch.save(
        state,
        cpt_file
      )

      print('Saved checkpoint file:', cpt_file)

    if train_loader is not None:
      net.eval()
      losses = []
      edge_losses = []
      node_losses = []
      for batchnum, batch in enumerate(test_loader):
        if maxbatches > 0 and batchnum >=maxbatches: break
        optimizer.zero_grad()
        batch.to(device)
        pred_n, pred_e = net(batch)
        loss_e = loss_fn(pred_e, batch.edge_label.reshape(len(batch.edge_label), 1).float())
        loss_n = loss_fn(pred_n, batch.y.reshape(len(batch.y), 1).float())

        loss = loss_n + loss_e
        #loss.backward()
        #optimizer.step()
        theloss = loss.item()
        running_loss += theloss
        if not batchnum % 10: print(f'{batchnum}: Loss: {theloss}')
        if not batchnum % 100 and batchnum > 0:
          print(f'\n(Batch {batchnum}) Loss: {running_loss / 100.}')
          running_loss = 0.
        losses.append(theloss)
        edge_losses.append(loss_e.item())
        node_losses.append(loss_n.item())
      epoch_test_losses.append(losses)
      epoch_test_node_losses.append(node_losses)
      epoch_test_edge_losses.append(edge_losses)



  fout.create_dataset('losses', data=np.array(epoch_losses))
  fout.create_dataset('edge_losses', data=np.array(epoch_edge_losses))
  fout.create_dataset('node_losses', data=np.array(epoch_node_losses))

  fout.create_dataset('test_losses', data=np.array(epoch_test_losses))
  fout.create_dataset('test_edge_losses', data=np.array(epoch_test_edge_losses))
  fout.create_dataset('test_node_losses', data=np.array(epoch_test_node_losses))

  print('Saved training file:', train_file)
  fout.close()


def make_trainers(args):
  net = EdgeConvNet(edge_convs=args.edge_convs, mid_features=args.mid_feats, do_mlp_op=args.mlp_op)
  loss_fn = torch.nn.BCELoss(reduction='mean')
  optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
  return net, optimizer, loss_fn

if __name__ == '__main__':
  
  parser = ap()
  parser.add_argument('--save', action='store_true')
  parser.add_argument('-i', required=True, type=str)
  parser.add_argument('--nepochs', type=int, default=1)
  parser.add_argument('--test', default=None)
  parser.add_argument('--max', default=-1, type=int)
  parser.add_argument('--lr', default=0.01, type=float)
  parser.add_argument('--edge_convs', default=[64, 128], type=int, nargs='+')
  parser.add_argument('--mid_feats', default=256, type=int)
  parser.add_argument('--mlp_op', action='store_true')

  args = parser.parse_args()

  train_data = ShowerFeatures(file_path=args.i)
  train_loader = GraphDataLoader(train_data,
                                 shuffle     = True,
                                 num_workers = 4,
                                 batch_size  = 64
                                )

  if args.test is not None:
    test_data = ShowerFeatures(file_path=args.test)
    test_loader = GraphDataLoader(test_data,
                                   shuffle     = True,
                                   num_workers = 4,
                                   batch_size  = 64
                                  )
  else:
    test_loader = None
  net, optimizer, loss_fn = make_trainers(args)

  train(net, optimizer, loss_fn, train_loader, test_loader=test_loader, save=args.save, maxbatches=args.max, nepochs=args.nepochs)
