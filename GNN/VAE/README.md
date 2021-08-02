# Auto-Encoding Variational Bayes - Pytorch
Paper introduces a stochastic variational inference and learning algorithm that scales to large datasets and, under some mild differentiability conditions, even works in the intractable case.  

![image](https://user-images.githubusercontent.com/37684658/127351477-91d37e1d-e774-4d9f-a87c-9cc6192efad2.png)
> Hwalseok Lee, Autoencoders : A way for Unsupervised Learning of Nonlinear Manifold


### Dataset  
MNIST

### Usage  
```bash
python train.py
```
> __Initial Configuration__  
> `epochs` : 20  
> `lr` : 0.001  
> `batch_size` : 100  
> `n_input` : 784  
> `n_hidden1` : 512  
> `n_hidden2` : 256  
> `n_z` : 2    

### Results  
```bash  
$ python train.py  
Epoch : 0/20 -- batch : 0/60000, loss : 549.7419
Epoch : 0/20 -- batch : 10000/60000, loss : 201.7692
Epoch : 0/20 -- batch : 20000/60000, loss : 182.6580
Epoch : 0/20 -- batch : 30000/60000, loss : 175.3173
Epoch : 0/20 -- batch : 40000/60000, loss : 179.5819
Epoch : 0/20 -- batch : 50000/60000, loss : 173.5274
==== Test loss: 170.4045
Epoch : 1/20 -- batch : 0/60000, loss : 172.8705
Epoch : 1/20 -- batch : 10000/60000, loss : 167.7758
Epoch : 1/20 -- batch : 20000/60000, loss : 167.4356
Epoch : 1/20 -- batch : 30000/60000, loss : 163.6354
Epoch : 1/20 -- batch : 40000/60000, loss : 159.5187
Epoch : 1/20 -- batch : 50000/60000, loss : 163.6039
==== Test loss: 162.1281
Epoch : 2/20 -- batch : 0/60000, loss : 154.4518
Epoch : 2/20 -- batch : 10000/60000, loss : 166.7407
Epoch : 2/20 -- batch : 20000/60000, loss : 154.5549
Epoch : 2/20 -- batch : 30000/60000, loss : 161.8107
Epoch : 2/20 -- batch : 40000/60000, loss : 163.5647
Epoch : 2/20 -- batch : 50000/60000, loss : 160.3686
==== Test loss: 157.9180
Epoch : 3/20 -- batch : 0/60000, loss : 161.2660
Epoch : 3/20 -- batch : 10000/60000, loss : 160.3248
Epoch : 3/20 -- batch : 20000/60000, loss : 154.4435
Epoch : 3/20 -- batch : 30000/60000, loss : 153.1563
Epoch : 3/20 -- batch : 40000/60000, loss : 151.6105
Epoch : 3/20 -- batch : 50000/60000, loss : 158.2676
...
```

### To Do  
- [X] Problem with loss value being nan during learning  
```bash
Epoch : 13/20 -- batch : 0/60000, loss : 149.1316
Epoch : 13/20 -- batch : 10000/60000, loss : 153.3385
Epoch : 13/20 -- batch : 20000/60000, loss : 143.8378
Epoch : 13/20 -- batch : 30000/60000, loss : 150.1471
Epoch : 13/20 -- batch : 40000/60000, loss : 147.8852
Epoch : 13/20 -- batch : 50000/60000, loss : 145.4127
==== Test loss: 147.6945
Epoch : 14/20 -- batch : 0/60000, loss : 152.2865
Epoch : 14/20 -- batch : 10000/60000, loss : 147.1944
Epoch : 14/20 -- batch : 20000/60000, loss : 143.4170
Epoch : 14/20 -- batch : 30000/60000, loss : 146.7341
Epoch : 14/20 -- batch : 40000/60000, loss : nan
Epoch : 14/20 -- batch : 50000/60000, loss : nan
==== Test loss: nan
Epoch : 15/20 -- batch : 0/60000, loss : nan
Epoch : 15/20 -- batch : 10000/60000, loss : nan
Epoch : 15/20 -- batch : 20000/60000, loss : nan
Epoch : 15/20 -- batch : 30000/60000, loss : nan
Epoch : 15/20 -- batch : 40000/60000, loss : nan
Epoch : 15/20 -- batch : 50000/60000, loss : nan
```  
SOLVED : 
If 0 is entered in torch.log(), it becomes `-inf` and causes a problem. Fixed the problem by adding eps
```python
def loss_function(x,x_hat,mu,sigma):
   eps = 1e-20 # nan 뜨는 문제 해결위해서 +1e-20 더해줌
   RCE = torch.sum(x.view(-1,784).mul(torch.log(x_hat+eps)) + (1 - x.view(-1,784)).mul(torch.log(1-x_hat+eps))) # binary cross entropy
   KLD = 0.5 * torch.sum(mu.pow(2) + sigma.pow(2) - torch.log(sigma.pow(2)+eps) - 1)
   ELBO = RCE - KLD
   loss = -ELBO
   return loss
```

- [ ] Try generating data using decoder

### References  
[1] Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." arXiv preprint arXiv:1312.6114 (2013)  
[2] [lyeoni/pytorch-mnist-VAE](https://github.com/lyeoni/pytorch-mnist-VAE)  
[3] [hwalsuklee/tensorflow-mnist-VAE](https://github.com/hwalsuklee/tensorflow-mnist-VAE)  
