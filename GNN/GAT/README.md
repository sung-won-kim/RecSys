# Graph Attention Networks

PyTorch implementation of Graph Attention Networks (GAT)  

### Requirements  

```bash
conda env create -f 'environment.yaml'
```

---  
### Contents  
![스크린샷 2021-07-07 오후 8 34 45](https://user-images.githubusercontent.com/37684658/124752527-ed8a8780-df62-11eb-8fb7-c86dc9776a4e.png)
![스크린샷 2021-07-07 오후 8 34 58](https://user-images.githubusercontent.com/37684658/124752563-f9764980-df62-11eb-99f2-05626109f395.png)


### Usage

```bash
cd GAT/src
python train.py
```  

```bash
cd GAT/src
python train.py --hidden 64 --epochs 100 --lr 0.05 --dropout 0.4
```

---  
### Dataset  
You can use 3 types of datasets.  
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
### Initial parameter  
`epochs` : 200  
`lr` : 0.005  
`weight_decay` : 0.0005  
`hidden` : 8  
`attention` : 8  
`dropout` : 0.6  

---
### References  
[AntonioLonga](https://github.com/AntonioLonga/PytorchGeometricTutorial.git)
