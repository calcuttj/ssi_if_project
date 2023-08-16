import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
import sys


if __name__ == '__main__':
  with h5.File(sys.argv[1], 'r') as fin:
    plt.plot(np.array(fin['losses']).flatten(), label='Total loss')
    plt.plot(np.array(fin['node_losses']).flatten(), label='Node loss')
    plt.plot(np.array(fin['edge_losses']).flatten(), label='Edge loss')
    plt.legend()
    plt.show()
