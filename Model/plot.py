import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser as ap


def plot_flat(fin):
  plt.plot(np.array(fin['losses']).flatten(), label='Total loss')
  plt.plot(np.array(fin['node_losses']).flatten(), label='Node loss')
  plt.plot(np.array(fin['edge_losses']).flatten(), label='Edge loss')

def plot_ave(fin):
  losses = np.array(fin['losses'])
  xs = [len(losses[0])]
  for i in range(1, len(losses)):
    xs.append(sum(xs) + len(losses[i]))
  xs = np.array(xs)
  plt.scatter(xs, np.mean(fin['losses'], axis=1), label='Total loss', marker='o')
  plt.scatter(xs, np.mean(fin['node_losses'], axis=1), label='Node loss', marker='o')
  plt.scatter(xs, np.mean(fin['edge_losses'], axis=1), label='Edge loss', marker='o')

if __name__ == '__main__':
  parser = ap()
  parser.add_argument('-i', required=True)
  parser.add_argument('--ave', action='store_true')
  args = parser.parse_args()

  with h5.File(args.i, 'r') as fin:
    if args.ave:
      plot_ave(fin)
    else:
      plot_flat(fin)
    plt.legend()
    plt.show()
