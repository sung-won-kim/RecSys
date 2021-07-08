from torch_geometric.datasets import Planetoid
# from torch_geometric.datasets import Amazon
import math
import torch

# ========================================
# load_data
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
# test accuracy
# ========================================

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)