import torch
import time
import numpy as np


class SVD():
   def __init__(self, train, test, k, lr, reg_param , epochs, verbose):
      self._train = train
      self._test = test
      self._k = k
      self._lr = lr
      self._epochs = epochs
      self._num_users, self._num_items = train.shape
      self._U = np.random.normal(0,0.1,size = (self._k, self._num_users))
      self._V = np.random.normal(0,0.1,size = (self._k, self._num_items))
      self._reg_param = reg_param
      self._verbose = verbose
      self._bu = np.zeros(self._num_users)
      self._bi = np.zeros(self._num_items)
      self._mu_train = self._train[np.where(self._train>0)].mean()

   def train(self):
      t = time.time()

      for epoch in range(self._epochs):
         for i in range(self._num_users):
            for j in range(self._num_items):
               if self._train[i,j] > 0 :
                  self.gradient_descent(i,j)

         train_loss, test_loss = self.RMSELoss()

         if self._verbose == True and ((epoch) % 10 == 0 ):
                  print("Iteration : %d, train_rsme = %.4f, test_rsme = %.4f, time : %.4f" % (epoch, train_loss, test_loss, time.time()-t))

      return self.whole_pred(), train_loss, test_loss

# ========================================
# U_transpose dot V
# ========================================
   def whole_pred(self):
      whole_pred = self._U.T.dot(self._V) + self._mu_train + self._bu[:,np.newaxis] + self._bi[np.newaxis,:]
      return whole_pred

   def pred(self,i,j):
      pred = self._U[:,i].T.dot(self._V[:,j]) + self._mu_train + self._bu[i] + self._bi[j]
      return pred

# ========================================
# I(indicator function) matrix는 조건문으로 반영해야함
# ========================================
   def gradient_descent(self,i,j):
      error = self._train[i,j] - self.pred(i,j)

      self._U[:,i] += self._lr * (error*self._V[:,j] - self._reg_param*self._U[:,i]) 
      self._V[:,j] += self._lr * (error*self._U[:,i] - self._reg_param*self._V[:,j]) 
      
      self._bu[i] += self._lr * (error - self._reg_param * self._bu[i] )
      self._bi[j] += self._lr * (error - self._reg_param * self._bi[j] )


# ========================================
# RSME
# ========================================
   def RMSELoss(self):
      train_loss = 0
      test_loss = 0
      xi, yi = self._train.nonzero()
      whole_pred = self.whole_pred()
      x_test, y_test = self._test.nonzero()

      for x, y in zip(xi, yi):
         train_loss += pow(self._train[x, y] - whole_pred[x, y], 2)

      for x, y in zip(x_test, y_test):
         test_loss += pow(self._test[x, y] - whole_pred[x, y], 2)

      return np.sqrt(train_loss/len(xi)), np.sqrt(test_loss/len(xi))