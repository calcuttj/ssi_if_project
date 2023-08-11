import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, EdgeConv

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(16, 32)
        self.conv2 = GCNConv(32, 64)
        self.conv3 = GCNConv(64, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return torch.sigmoid(x)

class EdgeConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.messageMap1 = nn.Sequential(nn.BatchNorm1d(32),
                                      nn.Linear(32, 128),
                                      nn.ReLU(),
                                      nn.BatchNorm1d(128),
                                      nn.Linear(128, 128),
                                      nn.ReLU(),
                                      nn.BatchNorm1d(128),
                                      nn.Linear(128, 128),
                                      nn.ReLU(),
                                      nn.BatchNorm1d(128),
                                      )
        self.conv1 = EdgeConv(self.messageMap1)

        self.messageMap2 = nn.Sequential(nn.Linear(256, 256),
                                      nn.ReLU(),
                                      nn.BatchNorm1d(256),
                                      nn.Linear(256, 256),
                                      nn.ReLU(),
                                      nn.BatchNorm1d(256),
                                      nn.Linear(256, 256),
                                      nn.ReLU(),
                                      nn.BatchNorm1d(256),
                                      )
        self.conv2 = EdgeConv(self.messageMap2)

        self.MLPhead = nn.Sequential(nn.Linear(256, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 256),
                                     nn.Linear(256, 1))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)

        x = self.MLPhead(x)

        return torch.sigmoid(x)

class EdgeConvNet_shallow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.messageMap1 = nn.Sequential(nn.BatchNorm1d(32),
                                         nn.Linear(32, 128),
                                         nn.ReLU(),
                                         nn.BatchNorm1d(128),
                                         nn.Linear(128, 128),
                                         nn.ReLU(),
                                        )
        self.conv1 = EdgeConv(self.messageMap1)

        self.messageMap2 = nn.Sequential(nn.Linear(256, 256),
                                         nn.ReLU(),
                                         nn.BatchNorm1d(256),
                                         nn.Linear(256, 256),
                                         nn.ReLU(),
                                        )
        self.conv2 = EdgeConv(self.messageMap2)

        self.MLPhead = nn.Sequential(nn.Linear(256, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 1))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)

        x = self.MLPhead(x)

        return torch.sigmoid(x)
