# HiGPPIM - A Hierarchical Graph Neural Network Framework for Predicting Protein-Protein Interaction Modulators with Functional Group Information and Hypergraph Structure

## 1. overview
![image](https://github.com/1zzt/PPII-AEAT/raw/main/overview.png)
## 2. Prerequisites
- networkx 3.1 
- numpy 1.23.5
- pandas 2.0.0  
- python 3.9.0
- pytorch 1.12.0
- rdkit 2022.9.5
- torch-cluster 1.6.0+pt112cu113
- torch-geometric 2.3.0
- torch-scatter 2.1.0+pt112cu113
- torch-sparse 0.6.16+pt112cu113
## 3. Datasets
We collected inhibitor and non-inhibitor data for nine different PPI families from Rodrigues’s work. These PPI families are **Bcl2-Like/Bak-Bax**, **Bromodomain/Histone**, **Cyclophilins**, **HIF-1a/p300**, **Integrins**, **LEDGF/IN**, **LFA/ICAM**, **Mdm2-Like/P53**, and **XIAP/Smac**.

We put the data for identification of PPI-specific small molecule inhibitors (classification task) on the `Datasets/classification` folder and put the data for quantitative prediction of inhibitory potency (regression task) on the `Datasets/regression` folder.

## 4. Usage
```
python main.py --dataset bcl2_bak --task regression --num_epochs 100 --batch_size 32 --lr 0.001 --gpu 0
```
