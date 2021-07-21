# Factorization Meets The Neighborhood: a Multifaceted Collaborative Filtering Model  
## SVD (Singular Value Decomposition  

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

### Reference  
[추천 알고리즘 - SVD (Singular Value Decomposition)](https://seing.tistory.com/67)
