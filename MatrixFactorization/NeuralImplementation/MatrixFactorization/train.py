import time
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utils import RMSELoss

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
                    help='learning rate (default = 0.01)')
parser.add_argument('--k', required = False, type=int, default=k,
                    help='# of k (default = 10)')
args = parser.parse_args()

# ========================================
# Data Load & Preprocessing
# ========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train = pd.read_csv("./data/user_10466_movies_5084_ratings.csv")
train = train[:300]
train = torch.FloatTensor(train.values).to(device)
num_user, num_item = train.shape
# print('here')
_U = Variable(torch.randn(num_user,args.k).to(device), requires_grad = True)
_V = Variable(torch.randn(num_item,args.k).to(device), requires_grad = True)

# ========================================
# Train
# ========================================
# print('here')
for epoch in range(args.epochs):
   # print('here')
   x = torch.matmul(_U,_V.transpose(0,1))
   # print('here')

   # print('here')
   # print(x.shape, train.shape)
   loss = RMSELoss(x,train)
   # print('here')
   loss.backward()
   # print('here')
   lr = args.lr
   # print('here')
   print(f'epoch : {epoch}, loss : {loss.item()}')

   with torch.no_grad():
         _U -= lr * _U.grad
         _V -= lr * _V.grad

         # 가중치 갱신 후에는 변화도를 직접 0으로 만듭니다.
         _U.grad = None
         _V.grad = None

print('final x')      
print(x)
print('original x')
print(train)


