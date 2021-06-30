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
  def __init__(self, dataset):
    super(GCNNet, self).__init__()
    self.conv1 = GCN(dataset.num_node_features, 16)
    self.conv2 = GCN(16, dataset.num_classes)

  def forward(self, data):
    x, edge_index = data.x, data.edge_index

    x = self.conv1(x, edge_index)
    x = F.relu(x)
    x = F.dropout(x, training=self.training)
    x = self.conv2(x, edge_index)

    return F.log_softmax(x, dim=1)
