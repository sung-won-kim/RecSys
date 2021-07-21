import numpy as np
import pandas as pd
from layers import occf
import torch
import torch.nn as nn
import torch.optim as optim
import data
import time
import argparse
import yaml

# ========================================
# configuration.yaml
# ========================================
with open('configuration.yaml') as f:
  conf = yaml.load(f)

epochs = conf['epochs']
k = conf['k']
reg_param = conf['reg_param']
alpha = conf['alpha']

# ========================================
# Training settings
# ========================================
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', required = False, type=int, default=epochs,
                    help='# of epochs (default = 50)')
parser.add_argument('--k', required = False, type=int, default=k,
                    help='v의 hyperparameter (default = 5)')
parser.add_argument('--reg_param', required = False, type=float, default=reg_param,
                    help='reg_param(default = 0.01)')   
parser.add_argument('--alpha', required = False, type = float, default=alpha,
                    help='alpha(default = 40)')   
parser.add_argument('--bottleneck', required = False, type = int, default= 0,
                    help='Bottleneck 개선 반영 (default = False)')   
parser.add_argument('--verbose', required = False, default=True,
                    help='verbose(default = True)')     
args = parser.parse_args()

# ========================================
# Load data & Preprocessing
# ========================================
train = data.train
test = data.test

# ========================================
# Load Model
# ========================================
occf = occf(train, test, args.k, args.epochs, args.reg_param, args.alpha, args.verbose, args.bottleneck)

# ========================================
# Train
# ========================================
occf.fit()
