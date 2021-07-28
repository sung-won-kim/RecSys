# prerequisites
import builtins
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from layers import VAE, loss_function
import yaml
import time
import argparse


# ========================================
# configuration.yaml
# ========================================
with open('configuration.yaml') as f:
  conf = yaml.load(f)

epochs = conf['epochs']
lr = conf['lr']
batch_size = conf['batch_size']
n_input = conf['n_input'] # 784
n_hidden1 = conf['n_hidden1'] # 512
n_hidden2 = conf['n_hidden2'] # 256
n_output =  conf['n_z'] # 2

# ========================================
# Training settings
# ========================================
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', required = False, type=int, default=epochs,
                    help='# of epochs (default = 20)')
parser.add_argument('--batch', required = False, type=int, default=batch_size,
                    help='# of batch_size (default = 100)')
parser.add_argument('--lr', required = False, type=float, default=lr,
                    help='learning rate (default = 0.001)')
parser.add_argument('--verbose', required = False, type = bool ,default=True,
                    help='verbose(default = True)')                    
                  
args = parser.parse_args()
# ========================================
# Load Data
# ========================================
# MNIST Dataset
train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)

# ========================================
# Data Loader
# ========================================
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch, shuffle=False)

# ========================================
# Load Model
# ========================================
model = VAE(n_input = n_input, n_hidden1 = n_hidden1, n_hidden2 = n_hidden2, n_output = n_output)
optimizer = optim.Adam(model.parameters(), lr = args.lr)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# ========================================
# Load Model
# ========================================
t = time.time()
for epoch in range(args.epochs):
   model.train()
   train_loss = 0
   for batch_idx, (data, _) in enumerate(train_loader):

      data = data.to(device)
      optimizer.zero_grad()

      x_hat, mu, sigma = model(data)
      loss = loss_function(data, x_hat, mu, sigma)

      loss.backward()
      train_loss += loss.item()
      optimizer.step()
      
      if args.verbose == True and batch_idx % 100 == 0:
         print(f'Epoch : {epoch}/{args.epochs} -- batch : {batch_idx * len(data)}/{len(train_loader.dataset)}, loss : {loss.item()/len(data):.4f}')

   model.eval()
   test_loss = 0
   with torch.no_grad():
      for data, _ in test_loader:
         data = data.to(device)
         x_hat, mu, sigma = model(data)

         test_loss += loss_function(data, x_hat, mu, sigma).item()

   test_loss /= len(test_loader.dataset)
   print(f'==== Test loss: {test_loss:.4f}')