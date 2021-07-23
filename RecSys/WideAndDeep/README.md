# Wide & Deep Learning for Recommender Systems
We present the Wide & Deep learning frame- work to achieve both memorization and generalization in one model, by jointly training a linear model component and a neural network component.  

![image](https://user-images.githubusercontent.com/37684658/126752432-92a6bd2b-4f5a-4712-8191-e24e5711cb89.png)  

### Dataset  
![image](https://user-images.githubusercontent.com/37684658/126752650-70671d0d-675f-4688-830f-990daac0f705.png)

* __The Deep Component__  
For categorical features, the original inputs are feature strings. Each of these sparse, high-dimensional categorical features are first converted into a low-demensional and dense real-valued vector (Embedding Vector)

* __The Wide Componet__  
The wide component is a generalized linear model. The feature set includes raw input features and transformed features (cross-product transformation)  
_e.g., `AND(gender = female, language = en)` 와 같이 2개씩 짝지어서 0, 1로 표현_

### Usage  
`data_wide.pkl` and `data_deep.pkl` files must be formed first using `data.py`.  
```bash
python data.py
```
then run  
```bash
python train.py
```
> __Initial Configuration__  
> epochs : 200  
> lr : 0.001  
> batch_size : 200  
> verbose : True  

### Results  
```bash  
$ python train.py --lr 0.01 --epoch 100
Iteration : 1 -- Total loss : 2330.3899, Test accuracy : 0.610, Time : 8.5785
Iteration : 10 -- Total loss : 2290.7583, Test accuracy : 0.630, Time : 82.8632
Iteration : 20 -- Total loss : 2282.7937, Test accuracy : 0.634, Time : 165.0813
Iteration : 30 -- Total loss : 2278.6594, Test accuracy : 0.636, Time : 247.7914
Iteration : 40 -- Total loss : 2275.7375, Test accuracy : 0.637, Time : 329.5415
Iteration : 50 -- Total loss : 2273.5178, Test accuracy : 0.638, Time : 411.3852
Iteration : 60 -- Total loss : 2271.7861, Test accuracy : 0.638, Time : 493.3249
Iteration : 70 -- Total loss : 2270.3079, Test accuracy : 0.639, Time : 575.4780
Iteration : 80 -- Total loss : 2269.0857, Test accuracy : 0.639, Time : 657.4634
Iteration : 90 -- Total loss : 2268.0269, Test accuracy : 0.640, Time : 738.7034
Iteration : 100 -- Total loss : 2267.1443, Test accuracy : 0.640, Time : 821.2824
```

### References  
[1] Cheng, Heng-Tze, et al. "Wide & deep learning for recommender systems." Proceedings of the 1st workshop on deep learning for recommender systems. 2016.
