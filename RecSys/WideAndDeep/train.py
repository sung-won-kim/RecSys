from model import WideAndDeep
import pickle
from sklearn.model_selection import train_test_split
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import WDDataset
from sklearn.metrics import roc_auc_score
import time
import argparse
import yaml


# ========================================
# configuration.yaml
# ========================================
with open('configuration.yaml') as f:
  conf = yaml.load(f)

epochs = conf['epochs']
lr = conf['lr']
batch_size = conf['batch_size']

# ========================================
# Training settings
# ========================================
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', required = False, type=int, default=epochs,
                    help='# of epochs (default = 200)')
parser.add_argument('--lr', required = False, type = float, default=lr,
                    help='learning rate (default = 0.001)')
parser.add_argument('--batch', required = False, type = int, default=batch_size,
                    help='lambda_v(default = 0.01)')
parser.add_argument('--verbose', required = False, default=True,
                    help='verbose(default = True)')                    
args = parser.parse_args()

# ========================================
# Load Dataset (created at data.py)
# ========================================
with open('data_deep.pkl', 'rb') as f:
    data_deep = pickle.load(f)
    
with open('data_wide.pkl', 'rb') as f:
    data_wide = pickle.load(f)

# ========================================
# Preprocessing data
# ========================================
_Y = data_wide.ratings
data_deep.drop(['ratings','userid','movieid'], axis=1, inplace = True)
data_wide.drop(['ratings','userid','movieid'], axis=1, inplace = True)

X_train_deep, X_test_deep = train_test_split(data_deep.values, test_size=0.3, random_state=100)
X_train_wide, X_test_wide = train_test_split(data_wide.values, test_size=0.3, random_state=100)
Y_train, Y_test = train_test_split(_Y, test_size=0.3, random_state=100)

X_train_deep_tensor = torch.LongTensor(X_train_deep)
X_train_wide_tensor = torch.FloatTensor(X_train_wide)
Y_train_tensor = torch.FloatTensor(Y_train.values)

X_test_deep_tensor = torch.LongTensor(X_test_deep)
X_test_wide_tensor = torch.FloatTensor(X_test_wide)
Y_test_tensor = torch.FloatTensor(Y_test.values)

data_train = WDDataset(X_train_wide_tensor, X_train_deep_tensor, Y_train_tensor)

# ========================================
# Load Model
# ========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = WideAndDeep(data_wide, data_deep).to(device)
optimizer = optim.Adagrad(model.parameters(), lr = args.lr)
criterion = nn.BCELoss()



# ========================================
# Train
# ========================================
t = time.time()
for epoch in range(args.epochs):
   train_loader = DataLoader(dataset = data_train, batch_size = args.batch, shuffle = True)
   model.train()
   total_loss = 0
   for batch_idx, batch in enumerate(train_loader):
      X_wide, X_deep, y = batch[0], batch[1], batch[2]
      X_wide, X_deep, y = X_wide.to(device), X_deep.to(device), y.to(device)

      optimizer.zero_grad()
      y_pred = model(X_wide,X_deep)
      loss = criterion(y_pred.squeeze(), y)
      loss.backward()
      optimizer.step()
      total_loss += loss
   
   model.eval()
   pred = model(X_test_wide_tensor.cuda(), X_test_deep_tensor.cuda())
   auc = roc_auc_score(Y_test, pred.cpu().detach().numpy())

   if(args.verbose == True):
      if(epoch % 1 == 0):
         print(f'Iteration : {epoch} -- Total loss : {total_loss:.4f}, Test accuracy : {auc:.3f}, Time : {time.time()-t:.4f}')