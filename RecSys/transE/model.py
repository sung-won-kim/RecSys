import numpy as np
from numpy.core.numeric import indices
import torch
import torch.nn as nn
# https://github.com/mklimasz/TransE-PyTorch/blob/master/model.py

class TransE(nn.Module):
   def __init__(self, data_df, num_entities, num_relations, device, k = 100, margin = 1.0, norm = 1):
      super(TransE, self).__init__()
      self.device = device
      self.dim = k
      self.norm = norm # L1
      self.dataset = data_df
      self.num_entities = num_entities
      self.num_relations = num_relations
      self.criterion = nn.MarginRankingLoss(margin = margin, reduction = 'none')
      self.entity_embed = self.entity_embedding()
      self.relation_embed = self.relation_embedding()

   # ========================================
   # init entity, relation embedding
   # ========================================
   def entity_embedding(self):
      entity_embedding = nn.Embedding(num_embeddings= self.num_entities, embedding_dim= self.dim)
      uniform_range = 6/np.sqrt(self.dim)
      entity_embedding.weight.data.uniform_(-uniform_range, uniform_range)
      return entity_embedding

   def relation_embedding(self):
      relation_embedding = nn.Embedding(num_embeddings=self.num_relations, embedding_dim= self.dim)
      uniform_range = 6/np.sqrt(self.dim)
      relation_embedding.weight.data.uniform_(-uniform_range, uniform_range)
      return relation_embedding

   # ========================================
   # MarginRankingLoss
   # ========================================
   def loss(self, positive_dt, negative_dt):
      target = torch.randn(1).sign().to(self.device) # margin 
      return self.criterion(positive_dt, negative_dt, target).requires_grad_()

   # ========================================
   # dissimilarity measure
   # ========================================
   def dissimilarity(self, triplets):
      heads = triplets[:,0]
      relations = triplets[:,1]
      tails = triplets[:,2]
      return (self.entity_embed(heads) + self.relation_embed(relations) - self.entity_embed(tails)).norm(p = self.norm, dim = 1)

   def forward(self, positive_triplets, negative_triplets):
      positive_dt = self.dissimilarity(positive_triplets)
      negative_dt = self.dissimilarity(negative_triplets)
      return self.loss(positive_dt, negative_dt), positive_dt, negative_dt

   # def predict(self, triplets , k = 10):
   #    heads = triplets[:,0]
   #    relations = triplets[:,1]
   #    tails = triplets[:,2]
   #    _, indices = torch.topk(self.dissimilarity(triplets), k, largest = False)
   #    return torch.sum(torch.eq(indices, tails)).item()