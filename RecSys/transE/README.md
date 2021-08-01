# Translating Embeddings for Modeling Multi-relational Data  
Bordes, Antoine, et al.  

### Pseudo code  
![image](https://user-images.githubusercontent.com/37684658/127763323-7fbd380f-433b-4e3e-bef8-0a71024c1780.png)

### Summary  
![image](https://user-images.githubusercontent.com/37684658/127763333-b3be7882-4341-4bd4-b42a-df2eb9e8a4e3.png)  
#### Optimizer  
![image](https://user-images.githubusercontent.com/37684658/127763408-ed5ec371-f80b-408a-a191-9ca57057ad48.png)  
#### Margin Ranking Loss  
![image](https://user-images.githubusercontent.com/37684658/127763478-b0c8e490-3002-48bc-bc4f-14e267b3be8d.png)


### Dataset  
WN18  

### Usage  
```bash 
python train.py
```
> __Initial Configuration__  
> __epochs__ : 20  
> __lr__ : 0.001  
> __batch_size__ : 200  
> __k__ : 100 (embedding dim.)
> __margin__ : 1.0  
> __norm__ : 1 (L1 : 1, L2 : 2)  
> __verbose__ : True  

### Results  
```bash
$ python train.py --lr 0.01
Epoch : 1/20 , loss : 2.2254, positive_dt : 60.2540, negative_dt : 60.2553, Time : 3.9198
Epoch : 2/20 , loss : 2.2662, positive_dt : 69.9218, negative_dt : 69.9388, Time : 7.8592
Epoch : 3/20 , loss : 2.2236, positive_dt : 76.4500, negative_dt : 76.4870, Time : 11.7956
Epoch : 4/20 , loss : 2.1815, positive_dt : 82.2498, negative_dt : 82.1759, Time : 15.7157
Epoch : 5/20 , loss : 2.1456, positive_dt : 86.1511, negative_dt : 86.1428, Time : 19.6453
Epoch : 6/20 , loss : 2.0828, positive_dt : 89.8940, negative_dt : 89.9384, Time : 23.5367
Epoch : 7/20 , loss : 2.0363, positive_dt : 94.9440, negative_dt : 94.9976, Time : 27.4146
Epoch : 8/20 , loss : 1.9457, positive_dt : 99.5129, negative_dt : 99.5506, Time : 31.3062
Epoch : 9/20 , loss : 1.9073, positive_dt : 103.0191, negative_dt : 103.0243, Time : 35.1981
Epoch : 10/20 , loss : 1.8307, positive_dt : 105.3752, negative_dt : 105.4105, Time : 39.0857
Epoch : 11/20 , loss : 1.8003, positive_dt : 106.9882, negative_dt : 107.0451, Time : 42.9832
Epoch : 12/20 , loss : 1.7581, positive_dt : 109.2325, negative_dt : 109.2798, Time : 46.9052
Epoch : 13/20 , loss : 1.7325, positive_dt : 111.3927, negative_dt : 111.4042, Time : 50.8279
Epoch : 14/20 , loss : 1.6929, positive_dt : 112.9998, negative_dt : 113.0360, Time : 54.7435
Epoch : 15/20 , loss : 1.6634, positive_dt : 114.8241, negative_dt : 114.8256, Time : 58.6578
Epoch : 16/20 , loss : 1.6282, positive_dt : 116.4050, negative_dt : 116.4118, Time : 62.5663
Epoch : 17/20 , loss : 1.6405, positive_dt : 117.2984, negative_dt : 117.3211, Time : 66.4783
Epoch : 18/20 , loss : 1.5994, positive_dt : 118.6522, negative_dt : 118.6661, Time : 70.3924
Epoch : 19/20 , loss : 1.5759, positive_dt : 119.2825, negative_dt : 119.2756, Time : 74.3156
Epoch : 20/20 , loss : 1.5635, positive_dt : 119.8673, negative_dt : 119.8570, Time : 78.1907
```

### Evaluation (TO DO)  
1. 하나의 head + relation에 모든 tail들에 대한 distance 산출
2. 구한 distance를 오름차순으로 정렬하여 ranking을 매기고, 실제 class (tail)의 rank를 구함
3. 실제 class의 rank가 10위 안에 드는 비율을 구하여 evaluation

### Discussion  
- [ ] positive distance는 줄어들고, negative distance는 늘어나는 방향으로 train 되어야 하지 않는 지?  
- [ ] evaluation 구현  

### References  
[1] Translating Embeddings for Modeling Multi-relational Data. Bordes, Antoine, et al.  
