import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, EdgeConv

import numpy as np
import os
from tqdm import tqdm

torch.multiprocessing.set_start_method('spawn')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

SEED=12345
_=np.random.seed(SEED)
_=torch.manual_seed(SEED)

from iftool.gnn_challenge import ShowerFeatures
from torch_geometric.loader import DataLoader as GraphDataLoader

from .nodeinference import *

def prepare_dataloader(path):
    data = ShowerFeatures(file_path = path)

    loader = GraphDataLoader(data,
                             shuffle     = True,
                             num_workers = 0,
                             batch_size  = 64
    )

    return loader

def trainLoop(model, dataloader, verbose = True):
    model.train()
    lossHistory = []
    if verbose:
        pbar = tqdm(dataloader)
        iterable = pbar
    else:
        iterable = dataloader
    
    for batch in iterable:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        target = batch.y.clone().detach().float()
        loss = F.binary_cross_entropy(out[:,0], target)

        if verbose:
            pbarMessage = " ".join(["loss:",
                                    str(round(loss.item(), 4))])
            pbar.set_description(pbarMessage)

        lossHistory.append(loss.item())
                        
        loss.backward()
        optimizer.step()

    return lossHistory

def testLoop(model, dataloader, verbose = True):
    model.eval()
    lossHistory = []
    if verbose:
        pbar = tqdm(dataloader)
        iterable = pbar
    else:
        iterable = dataloader

    for batch in iterable:
        batch = batch.to(device)
        out = model(batch)
        target = batch.y.clone().detach().float()
        loss = F.binary_cross_entropy(out[:,0], target)

        if verbose:
            pbarMessage = " ".join(["loss:",
                                    str(round(loss.item(), 4))])
            pbar.set_description(pbarMessage)

        lossHistory.append(loss.item())

    return lossHistory

def main(args):
    train_loader = prepare_dataloader(args.trainpath)
    test_loader = prepare_dataloader(args.testpath)

    # model = GCN().to(device)
    # model = EdgeConvNet().to(device)
    # model = EdgeConvNet_deep().to(device)
    model = EdgeConvNet_shallow().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1.e-2, weight_decay=5e-4)
    
    trainLossHistory = []
    testLossSummary = []

    from datetime import datetime
    timeNow = datetime.now()
    timestamp = '-'.join([''.join([str(timeNow.year),
                                   str(timeNow.month),
                                   str(timeNow.day),
                               ]),
                          ''.join([str(timeNow.hour),
                                   str(timeNow.minute),
                                   str(timeNow.second),
                               ]),
                      ])
    checkpointPath = 'checkpoints_'+timestamp
    os.mkdir(checkpointPath)

    for epoch in range(args.nEpochs):
        print ("epoch", epoch)
        print ("training...")
        epochalTrainLossHistory = trainLoop(model, train_loader, args.verbose)
        trainLossHistory.append(epochalTrainLossHistory)

        print ("testing...")
        model.eval()
        testLossSummary.append({})
        epochalTestLossHistory = testLoop(model, test_loader, args.verbose)

        testLossSummary[epoch]['mean'] = np.mean(epochalTestLossHistory)
        testLossSummary[epoch]['lq'] = np.quantile(epochalTestLossHistory, 0.16)
        testLossSummary[epoch]['uq'] = np.quantile(epochalTestLossHistory, 0.84)
        print (testLossSummary[epoch])

        os.mkdir(os.path.join(checkpointPath, 
                              str(epoch)))
        torch.save(dict(model = model.state_dict()), 
                   os.path.join(checkpointPath,
                                str(epoch),
                   'weights.ckpt'))
        np.savetxt(os.path.join(checkpointPath, 
                                str(epoch),
                                'trainLoss.dat'),
                   np.array(trainLossHistory[epoch]))
        np.savetxt(os.path.join(checkpointPath,
                                str(epoch),
                                'testLoss.dat'),
                   np.array([testLossSummary[epoch]['mean'],
                             testLossSummary[epoch]['lq'],
                   testLossSummary[epoch]['uq']]))
        
if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type = str,
                        default = 'data/if-graph-train.h5',
                        help = "input train data (hdf5)")
    parser.add_argument('--test', type = str,
                        default = 'data/if-graph-test.h5',
                        help = "input test data (hdf5)")

    parser.add_argument('-n', '--nEpochs', type = int,
                        default = 200,
                        help = "maximum number of epochs to train")
    parser.add_argument('-v', '--verbose',
                        action = 'store_true',
                        help = "print extra debug messages")
    
    args = parser.parse_args()

    main(args)
