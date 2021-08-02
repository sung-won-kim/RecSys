import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
   def __init__(self, n_input, n_hidden1, n_hidden2, n_output):
      super(VAE, self).__init__()

      self.n_input = n_input

      # encoder 
      self.fc1 = nn.Linear(n_input, n_hidden1)
      self.fc2 = nn.Linear(n_hidden1, n_hidden2)
      self.fc_mu = nn.Linear(n_hidden2, n_output)
      self.fc_sigma = nn.Linear(n_hidden2, n_output)

      # decoder
      self.fc3 = nn.Linear(n_output, n_hidden2)
      self.fc4 = nn.Linear(n_hidden2, n_input)

      self.dropout = nn.Dropout(p = 0.2, inplace = False)

   def gaussian_encoder(self, x):
      h = F.relu(self.fc1(x))
      # h = self.dropout(h)
      h = F.relu(self.fc2(h))
      return self.fc_mu(h), self.fc_sigma(h)

   def gaussian_sampling(self, mu, sigma):
      eps = torch.randn_like(sigma)
      return eps.mul(sigma.pow(2)).add_(mu)

   def bernoulli_decoder(self, z):
      h = F.relu(self.fc3(z))
      # h = self.dropout(h)
      return F.sigmoid(self.fc4(h))

   def forward(self, x):
      mu, sigma = self.gaussian_encoder(x.view(-1, self.n_input))
      z = self.gaussian_sampling(mu, sigma)
      return self.bernoulli_decoder(z), mu, sigma 

def loss_function(x,x_hat,mu,sigma):
   eps = 1e-20 # nan 뜨는 문제 해결위해서 +1e-20 더해줌
   RCE = torch.sum(x.view(-1,784).mul(torch.log(x_hat+eps)) + (1 - x.view(-1,784)).mul(torch.log(1-x_hat+eps))) # binary cross entropy
   KLD = 0.5 * torch.sum(mu.pow(2) + sigma.pow(2) - torch.log(sigma.pow(2)+eps) - 1)
   ELBO = RCE - KLD
   loss = -ELBO
   return loss

   



   

