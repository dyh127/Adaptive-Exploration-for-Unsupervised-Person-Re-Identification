#!/bin/bash

evnpath=$(dirname "$(dirname "$(pwd)")")
echo $evnpath

export CUDA_VISIBLE_DEVICES=4 


export PATH=$evnpath/env/gcc5/bin:$PATH
PYTHON=$evnpath/env/pytorch-py36-env/bin/python
export LD_LIBRARY_PATH=$evnpath/env/pytorch-py36-env/lib:$evnpath/env/gcc5/lib:$LD_LIBRARY_PATH
logpath=./checkpoint

xi=0.6
sourceset='duke'
targetset='market'
$PYTHON
#$PYTHON main.py -s ${sourceset} -t ${targetset} --resume checkpoint/AE_${sourceset}2${targetset}_xi_${xi}/checkpoint.pth.tar --data-dir  data/ --evaluate
