# Collaborative Filtering for Implicit Feedback Datasets (OCCF)  

### Preliminaries  
![image](https://user-images.githubusercontent.com/37684658/126501798-2ce27a43-ffd3-4497-bdb4-c1198a6207e4.png)  
![image](https://user-images.githubusercontent.com/37684658/126501830-983bea30-af81-49dc-abeb-b18886f74f73.png)


### Alternating-Least-Squares Optimization  
we alternate between re-computing user-factors and item-factors, and each step is guaranteed to lower value of the cost function.  
* updated user-factor (item-factors fixed)
![image](https://user-images.githubusercontent.com/37684658/126500881-8cd2f694-1f85-4fde-981b-c95641435332.png)
* updated item-factor (user-factors fixed)
![image](https://user-images.githubusercontent.com/37684658/126500921-153986b0-c133-4d72-95ab-1fcabfe2b362.png)  

### Evaluation Methodology  
![image](https://user-images.githubusercontent.com/37684658/126501959-20a1844a-e4c6-4b8c-8293-f681b88e3278.png)  
> r_ui^t : testset에서의 r_ui  
> rank_ui : 한 user에 대한 row에서 해당 item이 상위 몇 %에 rank하는 지


### Usage  
```bash
python train.py
```
> __Initial Configuration__  
> epochs : 50  
> reg_param : 0.01  
> k : 5  (latent factor의 hyperparameter)  
> verbose : True  
> alpha : 4  

### Results  
```bash  
$ python train.py
Iteration : 0, rank_bar = 0.3281, time : 4.6388
Iteration : 10, rank_bar = 0.0932, time : 46.6007
Iteration : 20, rank_bar = 0.0929, time : 88.0680
Iteration : 30, rank_bar = 0.0929, time : 129.4939
Iteration : 40, rank_bar = 0.0929, time : 171.0577
```

### Laboratory  
Goal : Solve the bottleneck problem  
* __Trial Option__  
```bash 
# Original Model
python train.py --bottleneck 0
```
```bash
# Application of expressions presented in the paper (non-zero is not yet implemented)
python train.py --bottleneck 1
```
```bash
# Application of expressions presented in the paper (non-zero is partially implemented)
python train.py --bottleneck 2
```  
* __Results__  
```bash  
$ python train.py --bottleneck 0 --epochs 20
Iteration : 0, rank_bar = 0.2667, time : 4.4775
```
```bash  
$ python train.py --bottleneck 1 --epochs 20
Iteration : 0, rank_bar = 0.2846, time : 15.3705
```
```bash
$ python train.py --bottleneck 2 --epochs 20
Iteration : 0, rank_bar = 0.2769, time : 11.4918
```
시간이 오히려 늘었다.

### References  
Hu, Yifan, Yehuda Koren, and Chris Volinsky. "Collaborative filtering for implicit feedback datasets." 2008 Eighth IEEE International Conference on Data Mining. Ieee, 2008.
