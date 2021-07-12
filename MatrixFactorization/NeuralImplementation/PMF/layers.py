import torch
import time
import numpy as np

class pmf():
   def __init__(self, train, test, k, lr,epochs,lambdau, lambdav, verbose):
      self._train = train
      self._test = test
      self._k = k
      self._lr = lr
      self._num_users, self._num_items = train.shape
      self._U = np.random.normal(0,0.1,size = (self._k,self._num_users))
      self._V = np.random.normal(0,0.1,size = (self._k,self._num_items))
      self._lambda_U = lambdau
      self._lambda_V = lambdav
      self._epochs = epochs
      self._verbose = verbose

   def train(self):
      t = time.time()

      for epoch in range(self._epochs):
         # print(self.epochs)
         for i in range(self._num_users):
            # print(self._num_users)
            for j in range(self._num_items):
               # print(self._num_items)
               if self._train[i,j] > 0 :
                  self.gradient_descent(i,j)
         # print("gradient done, time : %.4f" % (time.time()-t))

         train_loss, test_loss = self.RMSELoss()
         # print("loss done, time : %.4f" % (time.time()-t))

         if self._verbose == True and ((epoch) % 10 == 0 ):
                  print("Iteration : %d, train_rsme = %.4f, test_rsme = %.4f, time : %.4f" % (epoch, train_loss, test_loss, time.time()-t))

      return self.whole_pred(), train_loss, test_loss

# ========================================
# U_transpose dot V
# ========================================
   def whole_pred(self):
      pred = self._U.T.dot(self._V)
      return pred

   def pred(self,i,j):
      pred = self._U[:,i].T.dot(self._V[:,j])
      return pred

# ========================================
# I(indicator function) matrix는 조건문으로 반영해야함
# ========================================
   def gradient_descent(self,i,j):
      error = self._train[i,j] - self.pred(i,j)

      self._U[:,i] -= self._lr * (-error*self._V[:,j] + self._lambda_U*self._U[:,i]) 
      self._V[:,j] -= self._lr * (-error*self._U[:,i] + self._lambda_V*self._V[:,j]) 

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
