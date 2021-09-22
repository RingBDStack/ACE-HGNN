#!/usr/bin/env bash
export ACE_HOME=$(pwd)
export LOG_DIR="$ACE_HOME/logs"
export PYTHONPATH="$ACE_HOME:$PYTHONPATH"
export DATAPATH="$ACE_HOME/data"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64
source activate ace-hgnn  # replace with source hgcn/bin/activate if you used a virtualenv
