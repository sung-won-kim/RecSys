import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import data
from data import load_data
from model import TransE
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
k = conf['k'] # embedding dim.
margin = conf['margin']
norm = conf['norm'] #L1 : 1 or L2 : 

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
parser.add_argument('--k', required = False, type=int, default=k,
                    help='embedding dim (default = 100)')
parser.add_argument('--norm', required = False, type=int, default=k,
                    help='norm L1 or L2 (default = 1)')
parser.add_argument('--margin', required = False, type=float, default=margin,
                    help='margin (default = 1.0)')
parser.add_argument('--verbose', required = False, type = bool ,default=True,
                    help='verbose(default = True)')                    
                  
args = parser.parse_args()


# ========================================
# Load Data
# ========================================
data_df, data_tensor, train_df, train_torch, test_df, test_torch = load_data('./WN18/train.txt')
train_dataset = data.WB18Dataset(train_torch)
test_dataset = data.WB18Dataset(test_torch)
num_entities = len(np.unique(data_df[['head_word','tail_word']]))
num_relations = len(np.unique(data_df['relation']))

# ========================================
# Data Loader
# ========================================
train_loader = DataLoader(train_dataset, batch_size = args.batch)
test_loader = DataLoader(test_dataset, batch_size = args.batch)

# ========================================
# Load Model
# ========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransE(train_df, num_entities = num_entities, num_relations = num_relations, device = device)
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr = args.lr)

# ========================================
# train
# ========================================
t = time.time()
for epoch in range(args.epochs):
   model.train()
   total_loss = 0
   total_positive_dt = 0
   total_negative_dt = 0
   for batch_idx, batch in enumerate(train_loader):
      head, relation, tail = batch[0], batch[1], batch[2]
      head = head.to(device)
      relation = relation.to(device)
      tail = tail.to(device)

      positive_triplets = torch.stack((head,relation,tail), dim = 1)

      # negative triplets
      head_or_tail = torch.randint(high=2, size=head.size(), device = device)
      random_entities = torch.randint(high=len(np.unique(train_df.head_word)), size = head.size(), device = device)
      head_exchange = torch.where(head_or_tail == 1, random_entities, head)
      tail_exchange = torch.where(head_or_tail == 0, random_entities, tail)

      negative_triplets = torch.stack((head_exchange, relation, tail_exchange), dim = 1)


      loss, positive_dt, negative_dt = model(positive_triplets, negative_triplets)

      total_loss += loss.sum().item()
      total_positive_dt += positive_dt.sum().item()
      total_negative_dt += negative_dt.sum().item()

      optimizer.zero_grad()
      loss.sum().backward()
      
      optimizer.step()

   if args.verbose == True and epoch % 1 == 0:
      print(f'Epoch : {epoch+1}/{args.epochs} , loss : {total_loss/train_dataset.__len__():.4f}, positive_dt : {total_positive_dt/train_dataset.__len__():.4f}, negative_dt : {total_negative_dt/train_dataset.__len__():.4f}, Time : {time.time()-t:.4f}')
   
   # ========================================
   # test
   # ========================================
   # correct_test = 0
   # for batch_idx, batch in enumerate(test_loader):
   #    head, relation, tail = batch[0], batch[1], batch[2]
   #    head = head.to(device)
   #    relation = relation.to(device)
   #    tail = tail.to(device)
   #    triplets = torch.stack((head,relation,tail), dim = 1)
      
   #    # batch 중 하나의 head + relation 에 대해서 모든 tail (entities)들과 각각 distance 비교 후 오름차순 한 뒤, 실제 tail이 예측된 tail결과에서 몇 rank인 지 구함
   #    head_relation = torch.stack((head,relation), dim = 1) # [batch_size, 2]
   #    head_relation = torch.unsqueeze(head_relation, dim = 2) # total tail 넣을 공간 생성

   #    total_tails = np.unique(data_df.tail_word)

   #    predict_distance = torch.empty(num_entities)
      
   #    # head (batch_size)

   #    correct_test += model.predict(triplets, k = 10)
   # print(f"===>epoch {epoch+1}, test accuracy {correct_test/test_dataset.__len__()}")

