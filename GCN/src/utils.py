from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Amazon
import math
import torch

# ========================================
# Loading Dataset
# ========================================

def load_data(dataset_name):

  if dataset_name == 'cora':
    print("loading Cora dataset...")
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
  elif dataset_name == 'citeseer':
    print("loading CiteSeer dataset...")
    dataset = Planetoid(root='/tmp/Citeseer', name='CiteSeer')
  elif dataset_name == 'pubmed':
    print("loading PubMed dataset...")
    dataset = Planetoid(root='/tmp/Pubmed', name='PubMed')
  # elif dataset_name == 'amazon':
  #   print("loading Amazon dataset...")
  #   dataset = Amazon(root='/tmp/Amazon', name='Amazon')
  else :
    print("DATASET NOT FOUNDED (available dataset : cora, citeseer, pubmed)")
    quit()

  return dataset

# ========================================
# etc.
# ========================================

def uniform(size, tensor):
    if tensor is not None:
        bound = 1.0 / math.sqrt(size)
        tensor.data.uniform_(-bound, bound)


def kaiming_uniform(tensor, fan, a):
    if tensor is not None:
        bound = math.sqrt(6 / ((1 + a**2) * fan))
        tensor.data.uniform_(-bound, bound)


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def glorot_orthogonal(tensor, scale):
    if tensor is not None:
        torch.nn.init.orthogonal_(tensor.data)
        scale /= ((tensor.size(-2) + tensor.size(-1)) * tensor.var())
        tensor.data *= scale.sqrt()


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(1)


def normal(tensor, mean, std):
    if tensor is not None:
        tensor.data.normal_(mean, std)


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

# ========================================
# test accuracy
# ========================================

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)