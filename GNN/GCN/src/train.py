import time
import argparse
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_data, accuracy
import utils
from torch_geometric.datasets import Planetoid
from models import GCNNet
from tensorboardX import SummaryWriter

# ========================================
# configuration.yaml
# ========================================
with open('configuration.yaml') as f:
  conf = yaml.load(f)

epochs = conf['epochs']
lr = conf['lr']
weightDecay = conf['weight_decay']
num_hidden = conf['num_hidden']

# ========================================
# Training settings
# ========================================
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', required = False, type=int, default=epochs,
                    help='# of epochs (default = 200)')
parser.add_argument('--lr', required = False, type=float, default=lr,
                    help='learning rate (default = 0.01)')
parser.add_argument('--dataset', required = False, default='cora',
                    help='type of data (citeceer/cora/pubmed/nell) default = citeceer')
parser.add_argument('--weightdecay', required = False, type=float, default=weightDecay,
                    help='weight_decay (default = 5e-04)')
parser.add_argument('--hidden', required = False, type=int, default=num_hidden,
                    help='# of hidden (default = 16)')
args = parser.parse_args()

# ========================================
# tensorboard
# ========================================
writer = SummaryWriter(log_dir="runs/GCN_dataset({})_lr({})_epoch({})_hidden({})".format
                                    (args.dataset, args.lr, args.epochs, args.hidden))

# ========================================
# Load data
# ========================================
dataset = load_data(args.dataset)

# ========================================
# Load model
# ========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCNNet(dataset,args.hidden).to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weightdecay)

# ========================================
# Train
# ========================================
def train(epoch):
  t = time.time()
  model.train()
  optimizer.zero_grad()
  out = model(data)
  loss_train = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
  _, pred = model(data).max(dim=1)

  acc_train = accuracy(out[data.train_mask], data.y[data.train_mask])
  loss_train.backward()
  optimizer.step()

  model.eval()
  out = model(data)

  loss_val = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
  acc_val = accuracy(out[data.val_mask], data.y[data.val_mask])
  print('Epoch: {:04d}'.format(epoch),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

  return acc_train, loss_train, loss_val, acc_val

# ========================================
# Test
# ========================================
def test():
  model.eval()
  out = model(data)
  # _, pred = model(data).max(dim=1)
  # correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
  # acc = correct / int(data.test_mask.sum())
  acc = accuracy(out[data.test_mask], data.y[data.test_mask])
  print('Accuracy: {:.4f}'.format(acc))


# ========================================
# Implementation
# ========================================
def main():
  # Train model
  t_total = time.time()
  for epoch in range(args.epochs):
      acc_train, loss_train, loss_val, acc_val = train(epoch)
      writer.add_scalar('train_loss/epoch', loss_train.item(), epoch)   
      writer.add_scalar('train_acc/epoch', acc_train.item(), epoch)   
      writer.add_scalar('val_loss/epoch', loss_val.item(), epoch)   
      writer.add_scalar('val_acc/epoch', acc_val.item(), epoch)   
  print("Optimization Finished!")
  print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
  writer.close()
  # Testing
  test()

if __name__ =="__main__":
  main()

# ========================================
# tensorboard
# ========================================