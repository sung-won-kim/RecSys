import torch
import torch.nn.functional as F
import argparse
import yaml
from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import torch_geometric
from layers import GCN

# ========================================
# net
# ========================================
class GCNNet(torch.nn.Module):
  def __init__(self, dataset,num_hidden):
    super(GCNNet, self).__init__()
    self.conv1 = GCN(dataset.num_node_features, num_hidden)
    self.conv2 = GCN(num_hidden, dataset.num_classes)

  def forward(self, data):
    x, edge_index = data.x, data.edge_index

    x = self.conv1(x, edge_index)
    x = F.relu(x)
    x = F.dropout(x, training=self.training)
    x = self.conv2(x, edge_index)

    return F.log_softmax(x, dim=1)

