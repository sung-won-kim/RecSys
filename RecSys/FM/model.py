import torch
import torch.nn as nn

class FM(nn.Module):
   def __init__(self, num_feature, num_v_feature):
      # num_feature : dataset에서 변수의 개수 
      # num_v_feature : v 벡터의 feature, hyperparameter
      super(FM, self).__init__()
      self._w0 = nn.Parameter(torch.zeros(1))
      self._wi = nn.Parameter(torch.randn((num_feature)))
      self._vif = nn.Parameter(torch.randn((num_feature, num_v_feature)))
      

   def forward(self,input):
      # input = torch.FloatTensor(input)
      linear_terms = self._wi.matmul(input)
      interactions = 0.5 * ( sum( pow(input.matmul(self._vif),2) - pow(input,2).matmul(pow(self._vif,2)) ) )

      y_hat = torch.sigmoid(self._w0 + linear_terms + interactions)

      return y_hat