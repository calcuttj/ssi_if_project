import os
import numpy as np
import matplotlib.pyplot as plt

def main(args):
    checkpointSubDirs = [i for i in os.listdir(args.checkpointPath) if i != 'plots']
    checkpointSubDirs.sort(key = lambda x: int(x))

    trainLoss = [np.loadtxt(os.path.join(args.checkpointPath,
                                         thisCheckpointDir,
                                         'trainLoss.dat'))
                 for thisCheckpointDir in checkpointSubDirs]
    fullTrainLoss = np.concatenate(trainLoss)
    batchesPerEpoch = np.max([len(i) for i in trainLoss])
    epoch = np.arange(len(fullTrainLoss))/batchesPerEpoch
    plt.plot(epoch, fullTrainLoss, label = 'Train')

    testLoss = [np.loadtxt(os.path.join(args.checkpointPath,
                                        thisCheckpointDir,
                                        'testLoss.dat'))
                for thisCheckpointDir in checkpointSubDirs]
    testLossMean = np.array([thisEpochLoss[0] for thisEpochLoss in testLoss])
    testLossLQ = np.array([thisEpochLoss[1] for thisEpochLoss in testLoss])
    testLossUQ = np.array([thisEpochLoss[2] for thisEpochLoss in testLoss])
    epoch = np.arange(len(testLossMean)) + 1
    
    yErrLow = testLossMean - testLossLQ
    yErrHigh = testLossUQ - testLossMean
    plt.errorbar(epoch, testLossMean, 
                 yerr = (yErrLow, yErrHigh), 
                 fmt = 'o', 
                 label = 'Test')

    plt.semilogy()
    # plt.xlim(-1, 20)
    # plt.ylim(3.e-1, 2.e0)
    plt.legend(frameon = False)

    plotDir = os.path.join(args.checkpointPath,
                           'plots')
    if not os.path.exists(plotDir):
        os.mkdir(plotDir)
    plt.savefig(os.path.join(plotDir,
                             'loss.png'))
    
if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpointPath', type = str,
                        default = "",
                        help = "Root checkpoint directory")
    
    args = parser.parse_args()

    main(args)
