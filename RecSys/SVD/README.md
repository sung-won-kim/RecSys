# SVD (Singular Value Decomposition)

![스크린샷 2021-07-08 오후 11 56 06](https://user-images.githubusercontent.com/37684658/124944264-218ba880-e048-11eb-9cbf-dbe32296039c.png)  
> U : User와 latent factor 간의 관계  
> Sigma : diagonal matrix, 각 latent factor의 중요도  
> V : Item과 latent factor 간의 관계

1. 기존 행렬에 대해 null값을 0으로 채움  
2. U, sigma, V_transpose 행렬로 분해
3. latent facter matrix를 가지고 예측값을 생성하여 실제값과의 오차를 최소화하는 방식으로 진행  

### 목적함수  
- 기존의 평점인 r_ui와 x,y의 dot product으로 이루어진 r(hat)_ui간의 차이의 제곱합이 목적함수이며 이를 최소화 하는 방법으로 진행  
![스크린샷 2021-07-10 오후 10 57 05](https://user-images.githubusercontent.com/37684658/125165503-32682580-e1d2-11eb-802a-c5cd9fb958ba.png)
- 일반화 성능을 올리기 위하여 regularization term이 추가됨  
![스크린샷 2021-07-10 오후 10 57 10](https://user-images.githubusercontent.com/37684658/125165509-372cd980-e1d2-11eb-87c1-5c095964d888.png)
- 예측값과 실제값 사이의 error를 줄이기 위하여 bias term이 추가됨.  
> b_i : 아이템 i에서 전체 평균을 뺀 값들의 평균을 의미  
> b_u : 사용자 u에서 전체 평균을 뺀 값들의 평균을 의미  

![스크린샷 2021-07-10 오후 10 57 15](https://user-images.githubusercontent.com/37684658/125165514-3e53e780-e1d2-11eb-8982-55bf75f1598f.png)

### Usage  
```bash
python train.py
```

```bash
python train.py --epochs 200 --lr 0.01 --k 10 --regparam 0.01 --verbose True
```
> __Initial configuration__   
> `epochs` : 200  
> `lr` : 0.01  
> `k` : 5  
> `reg_param` : 0.01  

### Example  
```bash
$ python train.py --lr 0.01 --epochs 200
Iteration : 0, train_rsme = 0.9686, test_rsme = 0.3235, time : 1.2255
Iteration : 10, train_rsme = 0.8822, test_rsme = 0.3086, time : 13.5628
Iteration : 20, train_rsme = 0.8177, test_rsme = 0.3065, time : 25.9097
Iteration : 30, train_rsme = 0.7956, test_rsme = 0.3079, time : 38.2587
Iteration : 40, train_rsme = 0.7843, test_rsme = 0.3103, time : 50.6001
Iteration : 50, train_rsme = 0.7775, test_rsme = 0.3128, time : 63.0232
Iteration : 60, train_rsme = 0.7731, test_rsme = 0.3152, time : 75.4127
Iteration : 70, train_rsme = 0.7699, test_rsme = 0.3174, time : 87.7672
Iteration : 80, train_rsme = 0.7676, test_rsme = 0.3194, time : 100.1210
Iteration : 90, train_rsme = 0.7658, test_rsme = 0.3213, time : 112.4736
Iteration : 100, train_rsme = 0.7643, test_rsme = 0.3230, time : 124.8424
Iteration : 110, train_rsme = 0.7632, test_rsme = 0.3246, time : 137.1766
Iteration : 120, train_rsme = 0.7622, test_rsme = 0.3260, time : 149.5582
Iteration : 130, train_rsme = 0.7614, test_rsme = 0.3274, time : 161.9347
Iteration : 140, train_rsme = 0.7607, test_rsme = 0.3286, time : 174.3119
Iteration : 150, train_rsme = 0.7601, test_rsme = 0.3297, time : 186.6898
Iteration : 160, train_rsme = 0.7596, test_rsme = 0.3308, time : 199.0724
Iteration : 170, train_rsme = 0.7591, test_rsme = 0.3318, time : 211.4494
Iteration : 180, train_rsme = 0.7587, test_rsme = 0.3328, time : 223.8287
Iteration : 190, train_rsme = 0.7583, test_rsme = 0.3337, time : 236.2307
```


### Reference  
[추천 알고리즘 - SVD (Singular Value Decomposition)](https://seing.tistory.com/67)
