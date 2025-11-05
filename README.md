# GMV
Scaling Graph Inference by Serving Models as Views

This repository contains the code for the reproducibility of the experiments presented in the paper "Scaling Graph Inference by Serving Models as Views". 

This paper investigates a novel paradigm to scale graph inference by serving graph models as “views”. Given a class of local graph
models M, a graph model 𝑀, and a graph 𝐺, we aim to decide if it is possible to “approximate” the output 𝑀 over 𝐺 by only
referring to a set of local models in M, and their test cases. 

We introduce a class of memoization structures that package metadata of graph models, test data, and local inference as queryable graph
model views (GMVs). Based on GMVs, we study three fundamental problems: 
  (1) Inference, to approximate the output of 𝑀 by using GMVs with reduced time cost, and the conditions when this is possible; 
  (2) Selection, to decide whether a set of GMVs can be used to reconstruct the output of 𝑀, and if so, which to choose; and
  (3) Minimization, to minimize the storage cost of GMVs. For each problem, we establish doability and hardness results, and introduce efficient algorithms for    GMV-based inference analysis.

Using benchmark tasks, we experimentally verify that GMVs can significantly reduce the inference cost and scale graph inference to billion-scale graphs. We showcase the applications of GMV in cyber attack detection and bitcoin transaction anomaly detection.


## Organization of the code

All the code for the models described in the paper can be found in *codes/Cora.ipynb*, *codes/Arxiv.ipynb*, *codes/Yelp.ipynb*, *codes/Products.ipynb*, and *codes/MAG_Synthetic_Generator.ipynb*. 
We provide the datasets used in our experiments for users to validate our proposed methods located in: *Datasets.

## Prerequisites
Our code is based on Python3 (>= 3.11). The major libraries are listed as follows:
* NumPy (>= 1.26.3)
* Pandas (>= 2.0.1)
* Torch (>= 2.3.1)
* PyG (PyTorch Geometric) (>= 2.0.4)
* BisPy (>= 0.2.2)





