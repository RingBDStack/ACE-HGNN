ACE-HGNN in Pytorch
==================================================

This repository is the implementation of ACE-HGNN in PyTorch.

[[Paper] *ACE-HGNN: Adaptive Curvature Exploration Hyperbolic Graph Neural Network*](https://arxiv.org/pdf/2110.07888)

## Environment 
Below are some essential pypi packages we use on our own experiment environment. You may install their dependencies at the same time.
```
python==3.6.8
pytorch==1.6.0
nashpy==0.0.21
networkx==2.2
scikit-learn==0.20.3
numpy==1.16.2
pandas==0.24.2
scipy==1.2.1
```

## Usage 
### 1. Setup
* Clone this repo
* Create a virtual environment using conda or virtualenv.
  ```
  conda env create -f environment.yml
  virtualenv -p [PATH to python3.6 binary] ace-hgnn
  ```
* Enter the virtual environment and run `pip install -r requirements.txt`.

### 2. Usage
* Run `set_env.sh` in the command line. (Linux)
* Please refer to `config.py` for our Model's full parameters and their default values. 
* Run `python train.py [--param param_value]` to train our model, with setting custom parameters.
* Some training examples: 

|Dataset|Task|Command|Test AUC/F1|
|:---:|:---:|:---|:---:|
|Cora|LP|python -u train.py --task lp --dataset cora --model HGCN --lr 0.005 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.5 --weight-decay 0.001 --manifold PoincareBall --log-freq 5 --cuda 0 --c 1.0 --epochs 2000 --lr-reduce-freq 200 --lr-q 0.5|93.94|
|Cora|NC|python -u train.py --task nc --dataset cora --model HGCN --lr 0.01 --dim 16 --num-layers 2 --act leaky_relu --bias 1 --dropout 0.1 --weight-decay 0.0005 --manifold PoincareBall --log-freq 5 --cuda 0 --c 1.0 --epochs 2000 --lr-reduce-freq 200 --lr-q 0.5|81.80|
|PubMed|LP|python -u train.py --task lp --dataset pubmed --model HGCN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.5 --weight-decay 0.001 --manifold PoincareBall --log-freq 5 --cuda 0 --c 1.0 --lr-q 0.5 --save 1 --theta 0.5 --lr-reduce-freq None --gamma 0.2 --seed 4789|95.11|
|WebKB|LP|python -u train.py --task lp --dataset webkb --model HGCN --lr 0.005 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.5 --weight-decay 0.001 --manifold PoincareBall --log-freq 5 --cuda 0 --c 1.0 --epochs 2000 --lr-reduce-freq 200 --lr-q 0.5|94.47|
|PPI|LP|python -u train.py --task lp --dataset ppi --model HGCN --lr 0.0005 --dim 16 --num-layers 2 --act leaky_relu --bias 1 --dropout 0.05 --weight-decay 0.0005 --manifold PoincareBall --log-freq 5 --c 1.0 --epochs 5000 --lr-reduce-freq 500 --lr-q 0.5|92.03|
|PPI|NC|python -u train.py --task nc --dataset ppi --model HGCN --dim 16 --num-layers 2 --bias 1 --dropout 0.05 --weight-decay 0.0005 --manifold PoincareBall --log-freq 5 --c 1.0 --epochs 1000 --lr-reduce-freq 100 --lr-q 0.5 --lr 0.0005 --act tanh|67.54|




## Thanks
Some of the code was forked from the following repositories: 
* [HazyResearch/HGCN](https://github.com/HazyResearch/hgcn);  
* [SunQingYun1996/SUGAR](https://github.com/SunQingYun1996/SUGAR);  
* [tocom242242/nash_q_learning](https://github.com/tocom242242/nash_q_learning).  

We deeply thanks for their contributions to the open-source community. 

We also thanks for the computing infrastructure provided by Beijing Advanced Innovation Center for Big Data and Brain Computing (BDBC).

## Citation
```
@article{fu2021ace,
  title={ACE-HGNN: Adaptive Curvature Exploration Hyperbolic Graph Neural Network},
  author={Fu, Xingcheng and Li, Jianxin and Wu, Jia and Sun, Qingyun and Ji, Cheng and Wang, Senzhang and Tan, Jiajun and Peng, Hao and Yu, Philip S},
  journal={arXiv preprint arXiv:2110.07888},
  year={2021}
}
```

