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

## References  
Mnih, Andriy, and Russ R. Salakhutdinov. "Probabilistic matrix factorization." Advances in neural information processing systems. 2008.  
https://github.com/Namkyeong/RecSys_paper

