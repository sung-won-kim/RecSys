# import torch
import time
import argparse
import yaml
import data
import pandas as pd
# from tensorboardX import SummaryWriter
from layers import pmf

# ========================================
# configuration.yaml
# ========================================
with open('configuration.yaml') as f:
  conf = yaml.load(f)

epochs = conf['epochs']
lr = conf['lr']
k = conf['k']
lambda_U = conf['lambda_U']
lambda_V = conf['lambda_V']

# ========================================
# Training settings
# ========================================
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', required = False, type=int, default=epochs,
                    help='# of epochs (default = 200)')
parser.add_argument('--lr', required = False, type=float, default=lr,
                    help='learning rate (default = 0.01)')
parser.add_argument('--k', required = False, type=int, default=k,
                    help='k (default = 5)')
parser.add_argument('--lambdau', required = False, type=float, default=lambda_U,
                    help='lambda_u (default = 0.01)')
parser.add_argument('--lambdav', required = False, type=float, default=lambda_V,
                    help='lambda_v(default = 0.01)')
parser.add_argument('--verbose', required = False, default=True,
                    help='verbose(default = True)')                    
args = parser.parse_args()

# ========================================
# tensorboard
# ========================================
# writer = SummaryWriter(log_dir="runs/PMF_lr({})_epoch({})_k({})_lambda_U({})_lambda_V({})".format
#                                     (args.lr, args.epochs, args.k,args.lambdau, args.lambdav))

# ========================================
# Load data
# ========================================
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_data = data.train
# train_data = torch.FloatTensor(train_data)
# train_data.to(device)
test_data = data.test
# test_data = torch.FloatTensor(test_data)
# test_data.to(device)

# ========================================
# Load model
# ========================================
pmf = pmf(train_data, test_data, args.k, args.lr, args.epochs, args.lambdau, args.lambdav, args.verbose)

# ========================================
# Train
# ========================================

pmf.train()