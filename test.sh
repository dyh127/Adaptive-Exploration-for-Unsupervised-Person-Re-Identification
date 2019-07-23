export CUDA_VISIBLE_DEVICES=4 

xi=0.6
sourceset='duke'
targetset='market'
python main.py -s ${sourceset} -t ${targetset} --resume checkpoint/AE_${sourceset}2${targetset}_xi_${xi}/checkpoint.pth.tar --data-dir  data/ --evaluate
