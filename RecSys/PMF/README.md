# Probabilistic Matrix Factorization (PMF)
Implementation of Probabilistic Matrix Factorization (PMF) 

## Posterior Function  
![image](https://user-images.githubusercontent.com/37684658/125321409-9b3cd280-e377-11eb-9d2b-de9b40eba87f.png)

## Objective Function  
![image](https://user-images.githubusercontent.com/37684658/125320993-37b2a500-e377-11eb-97e0-9667f4922984.png)

## Usage  
```bash
python train.py
```
> initial configuration  
epochs : 200  
lr : 0.01  
k : 5  
lambda_U : 0.01  
lambda_V : 0.01  

## Example  
```bash
python train.py --epochs 100 --k 10 --lr 0.001 --lambdau 0.01 --lambdav 0.01
```

## Results  
```bash
$ python train.py --epochs 100 --k 10 --lr 0.001 --lambdau 0.01 --lambdav 0.01  
Iteration : 0, train_rsme = 3.6987, test_rsme = 1.2128, time : 1.1486
Iteration : 10, train_rsme = 1.9139, test_rsme = 0.8377, time : 12.6886
Iteration : 20, train_rsme = 1.1395, test_rsme = 0.5013, time : 24.2111
Iteration : 30, train_rsme = 1.0025, test_rsme = 0.3972, time : 35.7527
Iteration : 40, train_rsme = 0.9507, test_rsme = 0.3544, time : 47.3293
Iteration : 50, train_rsme = 0.9215, test_rsme = 0.3344, time : 58.9097
Iteration : 60, train_rsme = 0.9002, test_rsme = 0.3241, time : 70.4740
Iteration : 70, train_rsme = 0.8824, test_rsme = 0.3183, time : 82.0543
Iteration : 80, train_rsme = 0.8670, test_rsme = 0.3146, time : 93.6208
Iteration : 90, train_rsme = 0.8530, test_rsme = 0.3122, time : 105.2031
```
```bash
$ python train.py --epochs 100 --k 5 --lr 0.001 --lambdau 0.01 --lambdav 0.01
Iteration : 0, train_rsme = 3.6991, test_rsme = 1.2128, time : 1.1806
Iteration : 10, train_rsme = 2.0229, test_rsme = 0.8614, time : 12.7132
Iteration : 20, train_rsme = 1.1499, test_rsme = 0.5016, time : 24.2520
Iteration : 30, train_rsme = 1.0100, test_rsme = 0.3968, time : 35.8018
Iteration : 40, train_rsme = 0.9597, test_rsme = 0.3542, time : 47.3198
Iteration : 50, train_rsme = 0.9330, test_rsme = 0.3345, time : 58.8465
Iteration : 60, train_rsme = 0.9145, test_rsme = 0.3243, time : 70.3828
Iteration : 70, train_rsme = 0.9000, test_rsme = 0.3186, time : 81.9150
Iteration : 80, train_rsme = 0.8883, test_rsme = 0.3151, time : 93.4639
Iteration : 90, train_rsme = 0.8784, test_rsme = 0.3127, time : 105.0000
```
```bash
$ python train.py --epochs 100 --k 5 --lr 0.001 --lambdau 0.001 --lambdav 0.001
Iteration : 0, train_rsme = 3.6989, test_rsme = 1.2128, time : 1.1469
Iteration : 10, train_rsme = 1.8300, test_rsme = 0.8087, time : 12.7110
Iteration : 20, train_rsme = 1.1326, test_rsme = 0.4927, time : 24.2791
Iteration : 30, train_rsme = 1.0071, test_rsme = 0.3946, time : 35.8359
Iteration : 40, train_rsme = 0.9610, test_rsme = 0.3539, time : 47.4297
Iteration : 50, train_rsme = 0.9366, test_rsme = 0.3347, time : 59.0260
Iteration : 60, train_rsme = 0.9196, test_rsme = 0.3247, time : 70.6802
Iteration : 70, train_rsme = 0.9052, test_rsme = 0.3189, time : 82.5654
Iteration : 80, train_rsme = 0.8925, test_rsme = 0.3152, time : 94.7511
Iteration : 90, train_rsme = 0.8812, test_rsme = 0.3127, time : 106.9960
```
```bash
$ python train.py --epochs 100 --k 5 --lr 0.01 --lambdau 0.01 --lambdav 0.01
Iteration : 0, train_rsme = 2.5045, test_rsme = 0.8702, time : 1.2307
Iteration : 10, train_rsme = 0.8880, test_rsme = 0.3147, time : 13.8482
Iteration : 20, train_rsme = 0.8389, test_rsme = 0.3111, time : 25.9918
Iteration : 30, train_rsme = 0.8220, test_rsme = 0.3117, time : 38.1844
Iteration : 40, train_rsme = 0.8134, test_rsme = 0.3134, time : 50.4015
Iteration : 50, train_rsme = 0.8082, test_rsme = 0.3153, time : 61.9698
Iteration : 60, train_rsme = 0.8049, test_rsme = 0.3174, time : 73.4788
Iteration : 70, train_rsme = 0.8027, test_rsme = 0.3194, time : 84.9951
Iteration : 80, train_rsme = 0.8011, test_rsme = 0.3213, time : 96.5116
Iteration : 90, train_rsme = 0.7999, test_rsme = 0.3230, time : 108.0446
```
```bash
$ python train.py --epochs 100 --k 20 --lr 0.001 --lambdau 0.001 --lambdav 0.001
Iteration : 0, train_rsme = 3.6982, test_rsme = 1.2128, time : 1.2115
Iteration : 10, train_rsme = 1.7086, test_rsme = 0.7664, time : 12.7592
Iteration : 20, train_rsme = 1.1081, test_rsme = 0.4805, time : 24.2460
Iteration : 30, train_rsme = 0.9866, test_rsme = 0.3898, time : 35.6988
Iteration : 40, train_rsme = 0.9351, test_rsme = 0.3516, time : 47.1637
Iteration : 50, train_rsme = 0.9022, test_rsme = 0.3333, time : 58.6414
Iteration : 60, train_rsme = 0.8761, test_rsme = 0.3237, time : 70.1162
Iteration : 70, train_rsme = 0.8531, test_rsme = 0.3183, time : 81.5878
Iteration : 80, train_rsme = 0.8316, test_rsme = 0.3151, time : 93.0503
Iteration : 90, train_rsme = 0.8113, test_rsme = 0.3132, time : 104.5183
```

## References  
Mnih, Andriy, and Russ R. Salakhutdinov. "Probabilistic matrix factorization." Advances in neural information processing systems. 2008.  
https://github.com/Namkyeong/RecSys_paper

