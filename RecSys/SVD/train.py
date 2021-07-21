# import torch
import time
import argparse
import yaml
import data
import pandas as pd
# from tensorboardX import SummaryWriter
from model import SVD

# ========================================
# configuration.yaml
# ========================================
with open('configuration.yaml') as f:
  conf = yaml.load(f)

epochs = conf['epochs']
lr = conf['lr']
k = conf['k']
reg_param = conf['reg_param']

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
parser.add_argument('--regparam', required = False, type=float, default=reg_param,
                    help='regparam (default = 0.01)')
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
svd = SVD(train_data, test_data, args.k, args.lr, args.regparam, args.epochs, args.verbose)

# ========================================
# Train
# ========================================

svd.train()