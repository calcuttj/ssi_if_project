import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, EdgeConv

import numpy as np
import os
from tqdm import tqdm

import os
import numpy as np
import matplotlib.pyplot as plt

from training import 

torch.multiprocessing.set_start_method('spawn')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

SEED=12345
_=np.random.seed(SEED)
_=torch.manual_seed(SEED)

testpath = 'data/if-graph-test.h5'

from models import *

def make_scoreDistribution(labels, scores, plotDir):
    trueParentMask = labels == 1.
    trueChildMask = ~trueParentMask

    histkwargs = {'bins': np.linspace(0, 1, 100),
                  'histtype': 'step',
                  'density': True
    }

    plt.hist(scores[trueParentMask],
             label = 'True Parent Node',
             **histkwargs)
    plt.hist(scores[trueChildMask],
             label = 'True Child Node',
             **histkwargs)

    plt.legend(frameon = False)
    plt.semilogy()
    plt.xlabel(r'Network Score')

    plt.savefig(os.path.join(plotDir,
                             'scoreDistribution.png'))

def make_ROCcurve(labels, scores, plotDir):
    plt.figure()

    trueParentMask = labels == 1.
    trueChildMask = ~trueParentMask

    FPRlist = []
    TPRlist = []

    pbar = tqdm(np.linspace(0, 1, 201))
    for cutScore in pbar:
        inferredParent = scores > cutScore
        inferredChild = ~inferredParent

        TP = sum(np.logical_and(trueParentMask, inferredParent))
        FP = sum(np.logical_and(trueChildMask, inferredParent))
        TN = sum(np.logical_and(trueChildMask, inferredChild))
        FN = sum(np.logical_and(trueParentMask, inferredChild))
        # number of children identified as parents/total number of children
        FPR = FP/(FP + TN)
        # FPR = 0 
        # number of correctly identified parents/number of true parents
        TPR = TP/(TP + FN)
        # TPR = 0 

        pbarMessage = " ".join(["cut value:",
                                str(round(cutScore, 4)),
                                "FPR",
                                str(round(FPR, 4)),
                                "TPR:",
                                str(round(TPR, 4)),
                            ])
        pbar.set_description(pbarMessage)


        FPRlist.append(FPR)
        TPRlist.append(TPR)

    plt.plot(FPRlist, TPRlist)
    plt.xlabel(r'False Positive Rate')
    plt.ylabel(r'True Positive Rate')

    plt.savefig(os.path.join(plotDir,
                             'ROC.png'))

def main(args):
    test_loader = prepare_dataloader(testpath)
    
    model = EdgeConvNet().to(device)
    checkpoint = torch.load(args.checkpointPath,
                            map_location = device)
    model.load_state_dict(checkpoint['model'], strict = False)

    targets = np.empty((0,))
    scores = np.empty((0,))

    model.eval()
    pbar = tqdm(test_loader)
    for batch in pbar:
        batch = batch.to(device)
        out = model(batch)
        target = batch.y.clone().detach().float()
        loss = F.binary_cross_entropy(out[:,0], target)

        pbarMessage = " ".join(["loss:",
                                str(round(loss.item(), 4))])
        pbar.set_description(pbarMessage)
    
        targets = np.concatenate([targets, target.cpu().detach()])
        scores = np.concatenate([scores, out[:,0].cpu().detach()])


    parentDir = os.path.abspath(os.path.join(os.path.join(args.checkpointPath, 
                                                          os.path.pardir),
                                             os.path.pardir))
    plotDir = os.path.join(parentDir,
                           'plots')
    if not os.path.exists(plotDir):
        os.mkdir(plotDir)

    print ("Producing score distribution")
    make_scoreDistribution(targets, scores, plotDir)
    print ("Producing ROC curve")
    make_ROCcurve(targets, scores, plotDir)

if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpointPath', type = str,
                        default = "",
                        help = "Checkpoint directory")
    
    args = parser.parse_args()

    main(args)
