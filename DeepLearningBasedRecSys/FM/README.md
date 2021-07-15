# Factorization Machine (FM)
Implementation Binary Classification Model with 2-way Factorization Machine  

### Dataset  
__[Breast Cancer]('https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html')__

|Contents|Shape|
|---|---|
|data|(569,30)|
|target|(569,)|

### Usage  
```bash
# python train.py
```
> __Initial Configuration__  
> epochs : 200  
> lr : 0.001  
> k : 5  (v의 hyperparameter)  
> verbose : True  

### Example  
```bash
$ python train.py --epochs 300 --k 10 --lr 0.001
```

### Results  
```bash
$ python train.py --lr 0.001 --epochs 100 --k 5
Iteration : 0 -- Total loss : 565.6214, Test accuracy : 42.0, Time : 0.1539
Iteration : 10 -- Total loss : 158.3098, Test accuracy : 82.0, Time : 1.5311
Iteration : 20 -- Total loss : 119.2034, Test accuracy : 90.0, Time : 2.9257
Iteration : 30 -- Total loss : 103.6913, Test accuracy : 89.0, Time : 4.2949
Iteration : 40 -- Total loss : 94.7966, Test accuracy : 89.0, Time : 5.6861
Iteration : 50 -- Total loss : 88.6240, Test accuracy : 91.0, Time : 7.0527
Iteration : 60 -- Total loss : 83.9294, Test accuracy : 90.0, Time : 8.4349
Iteration : 70 -- Total loss : 80.1744, Test accuracy : 91.0, Time : 9.7746
Iteration : 80 -- Total loss : 77.0740, Test accuracy : 91.0, Time : 11.1041
Iteration : 90 -- Total loss : 74.4560, Test accuracy : 92.0, Time : 12.4500
```
|k|total loss|accuracy|time|  
|---|---|---|---|
|5|74.4560|94.3|12.4500|
|10|72.1391|94.2|13.5061|
|15|53.5665|95.8|14.0932|
|20|46.1793|95.6|14.8832|
|25|48.6053|95.1|15.9128|
|30|45.6153|94.7|16.3316|
> epochs : 100, lr : 0.001, average of 10 implementations

___"In sparse settings, typically a small k should be chosen because there is not enough data to estimate complex interactions W. Restricting k - and thus the expressiveness of the FM - leads to better generalization and thus improved interaction matrices under sparsity"___

### References  
Rendle, S. (2010, December). Factorization machines. In 2010 IEEE International conference on data mining (pp. 995-1000). IEEE.  
