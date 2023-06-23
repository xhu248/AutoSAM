#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/sam/lib


source activate pytorch

# data_dir=../vision_transformer/mmsegmentation/data/ade/ADEChallengeData2016/images/training
data_dir=../vision_transformer/mmsegmentation/data/coco_stuff10k/images/train2014
fold=2
tr_size=4
model_type=vit_h
dataset=ACDC
python scripts/inference.py --src_dir ../DATA/${dataset} -b 1 \
--data_dir ../DATA/${dataset}/imgs/ --save_dir ./output_experiment/sam_${model_type}_infer_${dataset}_f_${fold} \
--fold ${fold} --tr_size ${tr_size}  --model_type ${model_type} --gpu 2 \
 --dataset ${dataset} --num_classes 4


#fold=0
#tr_size=10
#python main_dist_seg.py --src_dir ../DATA/synapse --dist-url 'tcp://localhost:10001' \
#--data_dir ../DATA/synapse/imgs/ \
#--multiprocessing-distributed --world-size 1 --rank 0  \
#--fold ${fold} --tr_size ${tr_size}  --model_type ${model_type} \
#>output_log/sam_${model_type}_seg_synapse_f${fold}_tr_${tr_size}