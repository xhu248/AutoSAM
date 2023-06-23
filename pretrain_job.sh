#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/sam/lib

source activate pytorch

fold=0
dataset=ACDC
tr_size=100
python scripts/main_simclr.py --src_dir ../DATA/${dataset} \
--data_dir ../DATA/${dataset}/imgs/ --dataset ${dataset} \
--fold ${fold} --tr_size ${tr_size} -b 48 --contrast_mode byol


#fold=0
#tr_size=10
#dataset=ACDC
#python main_dist_seg.py --src_dir ../DATA/${dataset} --dist-url 'tcp://localhost:10001' \
#--data_dir ../DATA/${dataset}/imgs/ --save_dir ./sam_${model_type}_seg_${dataset}_f${fold}_tr_${tr_size} \
#--multiprocessing-distributed --world-size 1 --rank 0  -b 2 --dataset ${dataset} \
#--fold ${fold} --tr_size ${tr_size}  --model_type ${model_type} --num_classes 4 \
#>output_log/sam_${model_type}_seg_${dataset}_f${fold}_tr_${tr_size}

