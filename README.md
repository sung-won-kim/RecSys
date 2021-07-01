# 2021_DSAIL_summer_internship
---

## Graph Convolutional Networks in PyTorch

PyTorch implementation of Graph Convolutional Networks (GCNs) for semi-supervised classification.  

## Requirements  

```bash
conda env create -f 'environment.yaml'
```

## Usage

```bash
cd GCN/src
python train.py
```  

## 
```bash
cd GCN/src
python train.py --hidden 64 --epochs 100 --lr 0.05
```

## Dataset  
- CiteSeer
```bash  
python train.py --dataset citeseer
```  
- Cora
```bash  
python train.py --dataset citeseer
```  
- PubMed
```bash  
python train.py --dataset pubmed
```  

## Reference  
https://github.com/tkipf/pygcn  

