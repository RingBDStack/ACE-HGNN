ACE-HGNN: Adaptive Curvature Exploration Hyperbolic Graph Neural Network
==================================================

This repository is the implementation of ACE-HGNN in PyTorch. 

## Environment 
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
and their dependencies.

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
  * An example, for link prediction (LP) task on Cora dataset: `python train.py --task lp --dataset webkb --model HGCN --lr 0.005 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.5 --weight-decay 0.001 --manifold PoincareBall --log-freq 5 --cuda 0 --c 1.0`


## Thanks
Some of the code was forked from the following repositories: 
* [HazyResearch/HGCN](https://github.com/HazyResearch/hgcn);  
* [SunQingYun1996/SUGAR](https://github.com/SunQingYun1996/SUGAR);  
* [tocom242242/nash_q_learning](https://github.com/tocom242242/nash_q_learning).  

We deeply thanks for their contributions to the open-source community.

