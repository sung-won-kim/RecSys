import pandas as pd
import torch
import torch.nn as nn

def RMSELoss(yhat,y):
   loss = 0
   count = 0
   for i, j in y.nonzero():
      if count % 1000 == 0: 
         print(f'now count : {count}, total shape : ({len(y.nonzero())})')
      loss += pow(y[i,j] - yhat[i,j],2)
      count += 1
   return torch.sqrt(loss/count)
