# HiGPPIM - A Hierarchical Graph Neural Network Framework for Predicting Protein-Protein Interaction Modulators with Functional Group Information and Hypergraph Structure

## 1. overview
![image](https://github.com/1zzt/HiGppim/raw/main/overview.png)
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
For PPIM identification tasks, PPI families include **Bcl2-Like/Bak-Bax**, **Bromo Domain/Histone**, **CD4/gp120**, **LEDGF/IN**, **LFA/ICAM**, **Mdm2-Like/P53**, **Ras_SOS1**, and **XIAP/Smac**.
For PPIM potency prediction tasks, PPI families include **Bcl2-Like/Bak-Bax**, **Bromo Domain/Histone**, **CD4/gp120**, **LEDGF/IN**, **LFA/ICAM**, **Mdm2-Like/P53**, and **XIAP/Smac**.

## 4. Usage
```
python main.py --dataset bcl2_bak --task classification --batch_size 32 --lr 0.0001 --gpu 0
```
