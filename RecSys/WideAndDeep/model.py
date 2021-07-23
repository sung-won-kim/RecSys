import torch
import torch.nn as nn
import numpy as np

class WideAndDeep(nn.Module):

   def __init__(self, data_wide, data_deep):
      super(WideAndDeep, self).__init__()

      # Deep Model
      self.embed_gender = nn.Embedding(num_embeddings=len(np.unique(data_deep.gender)), embedding_dim=8)
      self.embed_age = nn.Embedding(num_embeddings=len(np.unique(data_deep.age)), embedding_dim=8)
      self.embed_occupation = nn.Embedding(num_embeddings=len(np.unique(data_deep.occupation)), embedding_dim=8)
      self.len_genres = len(data_deep.columns) - 3
      
      self.Linear_1 = nn.Linear(in_features = 24 + self.len_genres, out_features = 64)
      self.Linear_2 = nn.Linear(in_features = 64, out_features = 32)
      self.Linear_3 = nn.Linear(in_features = 32, out_features = 16)

      # WideAndDeep Model
      self.Linear = nn.Linear(in_features = len(data_wide.columns)+16, out_features = 1)
      self.Logistic = nn.Sigmoid()

   def forward(self, X_wide, X_deep):

      # Deep Model 
      embedding_genres = X_deep[:,:-3]
      embedding_gender = self.embed_gender(X_deep[:,-3])
      embedding_age = self.embed_age(X_deep[:,-2])
      embedding_occupation = self.embed_occupation(X_deep[:,-1])


      input_deep = torch.cat([embedding_genres, embedding_gender, embedding_age, embedding_occupation], dim = 1)

      input_deep = self.Linear_1(input_deep)
      input_deep = nn.ReLU()(input_deep)
      input_deep = self.Linear_2(input_deep)
      input_deep = nn.ReLU()(input_deep)
      input_deep = self.Linear_3(input_deep)
      output_deep = nn.ReLU()(input_deep)

      # WideAndDeep 
      input_wad = torch.cat([X_wide,output_deep], dim = 1)
      logits = self.Linear(input_wad)
      out = self.Logistic(logits)

      return out

   def init_weight(self):
      pass
      
