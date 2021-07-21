from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from model import FM
import torch
import torch.nn as nn
import torch.optim as optim
from utils import binary_acc
import time
import argparse
import yaml
from tensorboardX import SummaryWriter

# ========================================
# configuration.yaml
# ========================================
with open('configuration.yaml') as f:
  conf = yaml.load(f)

epochs = conf['epochs']
lr = conf['lr']
k = conf['k']

# ========================================
# Training settings
# ========================================
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', required = False, type=int, default=epochs,
                    help='# of epochs (default = 200)')
parser.add_argument('--lr', required = False, type=float, default=lr,
                    help='learning rate (default = 0.001)')
parser.add_argument('--k', required = False, type=int, default=k,
                    help='v의 hyperparameter (default = 5)')
parser.add_argument('--verbose', required = False, default=True,
                    help='verbose(default = True)')                    
args = parser.parse_args()

# ========================================
# tensorboard
# ========================================
writer = SummaryWriter(log_dir="runs/FM_lr({})_epoch({})_k({})".format
                                    (args.lr, args.epochs, args.k))

# ========================================
# Load data & Preprocessing
# ========================================
scaler = MinMaxScaler()
file = load_breast_cancer()
X, Y = torch.FloatTensor(file['data']), torch.FloatTensor(file['target'])
X = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)

_, num_feature= X.shape
num_v_feature = args.k

# ========================================
# Load Model
# ========================================
model = FM(num_feature, num_v_feature)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# ========================================
# GPU ==> 왜 cpu가 더 빠르지..?
# ========================================
model.to(device)
X_train = torch.FloatTensor(X_train).to(device)
X_test = torch.FloatTensor(X_test).to(device)
Y_train = torch.FloatTensor(Y_train).to(device)
Y_test =torch.FloatTensor(Y_test).to(device)

# ========================================
# Train
# ========================================
t = time.time()
for epoch in range(args.epochs):
   total_loss = 0
   for input, y in zip(X_train, Y_train):
      optimizer.zero_grad()
      y_pred = model(input)
      loss = criterion(y_pred, y.reshape(1))
      loss.backward()
      total_loss += loss
      optimizer.step()

   y_pred = torch.zeros_like(Y_test)
   for i, input in enumerate(X_test):
      y_pred[i] = model(input)
   acc = binary_acc(y_pred,Y_test).item()
   writer.add_scalar('total_loss/epoch', total_loss.item(), epoch)
   writer.add_scalar('test_acc/epoch', acc, epoch)
   writer.add_scalar('time/epoch', time.time()-t, epoch)

   if(args.verbose == True):
      if(epoch % 10 == 0):
         print(f'Iteration : {epoch} -- Total loss : {total_loss:.4f}, Test accuracy : {acc}, Time : {time.time()-t:.4f}')
      
   writer.close()




