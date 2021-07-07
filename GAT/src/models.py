from torch_geometric.nn.conv import GATConv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GAT(torch.nn.Module):
  def __init__(self, dataset, hid, in_head, dropout_p):
    super(GAT, self).__init__()
    self.hid = 8 # layer
    self.in_head = 8 # attention
    self.out_head = 1 
    self.dropout_p = dropout_p

    self.conv1 = GATConv(dataset.num_features, self.hid, heads=self.in_head, dropout = dropout_p)
    self.conv2 = GATConv(self.hid*self.in_head, dataset.num_classes, concat = False, heads=self.out_head, dropout= dropout_p)

  def forward(self, data):
    x, edge_index = data.x, data.edge_index
            
    x = F.dropout(x, p= self.dropout_p, training=self.training)
    x = self.conv1(x, edge_index)
    x = F.elu(x)
    x = F.dropout(x, p= self.dropout_p, training=self.training)
    x = self.conv2(x, edge_index)
    
    return F.log_softmax(x, dim=1)