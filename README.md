# Towards Unsupervised Deep Graph Structure Learning

This is the source code of paper Towards Unsupervised Deep Graph Structure Learning (SUBLIME). 

![The proposed framework](pipeline.png)

## REQUIREMENTS
This code requires the following:
* Python==3.7
* PyTorch==1.7.1
* DGL==0.7.1
* Numpy==1.20.2
* Scipy==1.6.3
* Scikit-learn==0.24.2
* Munkres==1.1.4
* ogb==1.3.1

## USAGE
### Step 1: All the scripts are included in the "scripts" folder. Please get into this folder first.
```
cd scripts
```

### Step 2: Run the experiments you want:

\[Cora\]Node classification @ structure inference:
```
bash cora_si.sh
```
\[Cora\]Node classification @ structure refinement:
```
bash cora_sr.sh
```
\[Cora\]Node clustering @ structure refinement:
```
bash cora_clu.sh
```
\[Citeseer\]Node classification @ structure inference:
```
bash citeseer_si.sh
```
\[Citeseer\]Node classification @ structure refinement:
```
bash citeseer_sr.sh
```
\[Citeseer\]Node clustering @ structure refinement:
```
bash citeseer_clu.sh
```
\[Pubmed\]Node classification @ structure inference:
```
bash pubmed_si.sh
```
\[Pubmed\]Node classification @ structure refinement:
```
bash pubmed_sr.sh
```

