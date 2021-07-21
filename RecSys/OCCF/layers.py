import torch
import time
import numpy as np

class occf():
   def __init__(self, train, test, k ,epochs,reg_param, alpha, verbose, bottleneck):
      self._train = train
      self._test = test
      self._k = k
      self._alpha = alpha
      self._num_users, self._num_items = train.shape
      self._U = np.random.normal(0,0.1,size = (self._num_users,self._k))
      self._V = np.random.normal(0,0.1,size = (self._num_items,self._k))
      self._reg_param = reg_param
      self._P = np.array(np.vectorize(lambda x: 0 if x==0 else 1)(train), dtype = np.float64)
      self._C = 1 + self._alpha * self._train
      self._epochs = epochs
      self._verbose = verbose
      self._bottleneck = bottleneck

   def fit(self):
      t = time.time()

      if(self._bottleneck == 0):
         for epoch in range(self._epochs):
            # t0 = time.time()
            self.als_user()
            # print("als_user time : %.4f" % (time.time()-t0))
            # t1 = time.time()
            self.als_item()
            # print("als_item time : %.4f" % (time.time()-t1))
            # t2 = time.time()
            rank_bar = self.rank_bar()
            # print("rank_bar time : %.4f" % (time.time()-t2))

            if self._verbose == True and ((epoch) % 10 == 0 ):
               print("Iteration : %d, rank_bar = %.4f, time : %.4f" % (epoch, rank_bar, time.time()-t))

      if(self._bottleneck == 1):
         for epoch in range(self._epochs):
            t0 = time.time()
            y_t_y_item = self._V.T.dot(self._V)
            # print("y_t_y_user time : %.4f" % (time.time()-t0))
            # t0 = time.time()
            self.als_user_bn(y_t_y_item)
            # print("als_user_bn time : %.4f" % (time.time()-t0))
            # t0 = time.time()
            y_t_y_user = self._U.T.dot(self._U)
            # print("y_t_y_item time : %.4f" % (time.time()-t0))
            # t0 = time.time()
            self.als_item_bn(y_t_y_user)
            # print("als_item_bn time : %.4f" % (time.time()-t0))
            rank_bar = self.rank_bar()
            
            if self._verbose == True and ((epoch) % 10 == 0 ):
               print("Iteration : %d, rank_bar = %.4f, time : %.4f" % (epoch, rank_bar, time.time()-t))
            
      if(self._bottleneck == 2):
         for epoch in range(self._epochs):
            t0 = time.time()
            y_t_y_item = self._V.T.dot(self._V)
            # print("y_t_y_user time : %.4f" % (time.time()-t0))
            # t0 = time.time()
            self.als_user_bn_edited(y_t_y_item)
            # print("als_user_bn time : %.4f" % (time.time()-t0))
            # t0 = time.time()
            y_t_y_user = self._U.T.dot(self._U)
            # print("y_t_y_item time : %.4f" % (time.time()-t0))
            # t0 = time.time()
            self.als_item_bn_edited(y_t_y_user)
            # print("als_item_bn time : %.4f" % (time.time()-t0))
            rank_bar = self.rank_bar()

            if self._verbose == True and ((epoch) % 10 == 0 ):
               print("Iteration : %d, rank_bar = %.4f, time : %.4f" % (epoch, rank_bar, time.time()-t))
# ========================================
# Prediction
# ========================================
   def pred(self):
      pred = self._U.dot(self._V.T)
      return pred

# ========================================
# ALS (bottleneck 반영 X)
# ========================================

   def als_user(self):
      V_t = np.transpose(self._V)
      for m in range(self._num_users):
         C_u = np.diag(self._C[m,:])
         part_1 = np.linalg.inv(V_t.dot(C_u).dot(self._V)+self._reg_param * np.identity(self._k))
         part_2 = V_t.dot(C_u).dot(self._P[m])
         self._U[m,:] = np.dot(part_1,part_2)

   def als_item(self):
      U_t = np.transpose(self._U)
      for n in range(self._num_items):
         C_i = np.diag(self._C[:,n])
         part_1 = np.linalg.inv(U_t.dot(C_i).dot(self._U)+self._reg_param * np.identity(self._k))
         part_2 = U_t.dot(C_i).dot(self._P[:,n])
         self._V[n,:] = np.dot(part_1,part_2)

# ========================================
# ALS (bottleneck 반영)
# ========================================
   def als_user_bn(self,y_t_y):
      V_t = np.transpose(self._V)
      for m in range(self._num_users):
         C_u = np.diag(self._C[m,:])
         part_1 = np.linalg.inv(y_t_y + V_t.dot(C_u-np.identity(self._num_items)).dot(self._V)+self._reg_param * np.identity(self._k))
         part_2 = V_t.dot(C_u).dot(self._P[m])
         self._U[m,:] = np.dot(part_1,part_2)

   def als_item_bn(self,y_t_y):
      U_t = np.transpose(self._U)
      for n in range(self._num_items):
         C_i = np.diag(self._C[:,n])
         part_1 = np.linalg.inv(y_t_y + U_t.dot(C_i-np.identity(self._num_users)).dot(self._U)+self._reg_param * np.identity(self._k))
         part_2 = U_t.dot(C_i).dot(self._P[:,n])
         self._V[n,:] = np.dot(part_1,part_2)

   def als_user_bn_edited(self,y_t_y):
      V_t = np.transpose(self._V)
 
      for m in range(self._num_users):

         C_u = np.diag(self._C[m,:])
         v_t_c = np.zeros((self._k, self._num_items))
         v_t_c_i = np.zeros_like(v_t_c)
         for i in range(self._k):
            for j in range(self._num_items):
               v_t_c[i,j] = V_t[i,j] * (C_u[j,j])
               v_t_c_i[i,j] = V_t[i,j] * (C_u[j,j] - 1)

         part_1 = np.linalg.inv(y_t_y + v_t_c_i.dot(self._V) + self._reg_param * np.identity(self._k))
         part_2 = v_t_c.dot(self._P[m])

         self._U[m,:] = np.dot(part_1,part_2)
   
   def als_item_bn_edited(self,y_t_y):
      U_t = np.transpose(self._U)
 
      for n in range(self._num_items):

         C_i = np.diag(self._C[:,n])
         u_t_c = np.zeros((self._k, self._num_users))
         u_t_c_i = np.zeros_like(u_t_c)
         for i in range(self._k):
            for j in range(self._num_users):
               u_t_c[i,j] = U_t[i,j] * (C_i[j,j])
               u_t_c_i[i,j] = U_t[i,j] * (C_i[j,j] - 1)

         part_1 = np.linalg.inv(y_t_y + u_t_c_i.dot(self._U) + self._reg_param * np.identity(self._k))
         part_2 = u_t_c.dot(self._P[:,n])
         
         self._V[n,:] = np.dot(part_1,part_2)

   # def als_user_bn(self,y_t_y):
   #    V_t = np.transpose(self._V)
   #    for i in range(self._num_users):
   #       for j in range(self._num_items):

   #       C_u = np.diag(self._C[m,:])
   #       part_1 = np.linalg.inv(y_t_y + V_t.dot(C_u-np.identity(self._num_items)).dot(self._V)+self._reg_param * np.identity(self._k))
   #       part_2 = V_t.dot(C_u).dot(self._P[m])
   #       self._U[m,:] = np.dot(part_1,part_2)
# ========================================
# Rank bar
# ========================================
   def rank_bar(self):
      pred = self.pred()
      tmp1 = 0
      tmp2 = 0
      
      for i in range(self._num_users):
         rank_index = pred[i].argsort()[::-1]
         for rank, j in enumerate(rank_index):
            tmp1 += self._test[i,j]*(rank/self._num_items)
            tmp2 += self._test[i,j]

      rank_bar = tmp1/tmp2

      return rank_bar