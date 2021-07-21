# Recommendations via Optimization  
__Goal__ : Make good recommendations  
- Quantify goodness using __RMSE__ : __Lower RMSE → better recommendations__  
- Let's set build a system such that it works well on known (user, item) ratings  
  And hope the system will also predict well the unknown ratings.   

---

## ContentBasedFiltering.py  

Implementation of basic level of content based filtering.  
CosineSimilarity(__genres__)  

---  

## ItemBasedCollaborativeFiltering.py   

Implementation of basic level of Item based collaborative filtering.  
CosineSimilarity(__item에 대한 user들의 ratings__)  

---  

## MatrixFactorization.py  
Implementation of basic level of MatrixFactorization  
using SVD(Singualr Value Decomposition) with Pearson Correlation Coefficient  
___item(영화)와 비슷한 items(영화들)을 찾아주는 목적___  

---   

## MatrixFactorization2.py  
Implementation of basic level of MatrixFactorization  
using SVD(Singualr Value Decomposition)  
___user(사용자)에게 items(영화)를 추천해주는 목적___  

---
## SVD (Singular Value Decomposition)  
Implementation of SVD  
