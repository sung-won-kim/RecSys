# Graph Convolutional Networks in PyTorch

PyTorch implementation of Graph Convolutional Networks (GCNs) for semi-supervised classification.  

### Requirements  

```bash
conda env create -f 'environment.yaml'
```
---  
### Contents  
![스크린샷 2021-07-07 오전 10 24 59](https://user-images.githubusercontent.com/37684658/124686175-973f2980-df0d-11eb-8ba5-049172794e3d.png)
![스크린샷 2021-07-07 오전 10 25 18](https://user-images.githubusercontent.com/37684658/124686204-a1612800-df0d-11eb-9e3e-265850b04a14.png)
![스크린샷 2021-07-07 오전 10 25 25](https://user-images.githubusercontent.com/37684658/124686215-a58d4580-df0d-11eb-8ade-e7001073e65a.png)
![스크린샷 2021-07-07 오전 10 26 35](https://user-images.githubusercontent.com/37684658/124686286-cf466c80-df0d-11eb-977b-ed6f19e363e8.png)

---  

### Limitations and Future Work
![스크린샷 2021-07-07 오전 10 26 11](https://user-images.githubusercontent.com/37684658/124686267-c190e700-df0d-11eb-89c3-97a520bd9134.png)  
> __In all of the spectral approaches, the learned filters depend on the Laplacian eigenbasis, which depends on the graph structure. Thus, a model trained on a specific structure can not be directly applied to a graph with a different structure.__  
> Velicˇkovic ́, Petar, et al. "Graph attention networks." arXiv preprint arXiv: 1710.10903 (2017)

--- 

### Usage

```bash
cd GCN/src
python train.py
```  

### 
```bash
cd GCN/src
python train.py --hidden 64 --epochs 100 --lr 0.05
```
---  
### Dataset  
- CiteSeer
```bash  
python train.py --dataset citeseer
```  
- Cora
```bash  
python train.py --dataset cora
```  
- PubMed
```bash  
python train.py --dataset pubmed
```  
---  
### Reference  
https://github.com/tkipf/pygcn  

