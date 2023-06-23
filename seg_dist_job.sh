#!/bin/bash


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/sam/lib

source activate pytorch

# model_path=./save/simclr_ACDC/b_48_f0_model.pth
fold=0
gpu=2
tr_size=15
model_type=vit_b
dataset=ACDC
output_dir=sam_${model_type}_seg_${dataset}_f${fold}_tr_${tr_size}_dec
python main_dist_seg.py --src_dir ../DATA/${dataset} \
--data_dir ../DATA/${dataset}/imgs/ --save_dir ./${output_dir}  \
--b 4 --dataset ${dataset} --gpu ${gpu} \
--fold ${fold} --tr_size ${tr_size}  --model_type ${model_type} --num_classes 4

tr_size=20
output_dir=sam_${model_type}_seg_${dataset}_f${fold}_tr_${tr_size}_dec
python main_dist_seg.py --src_dir ../DATA/${dataset} \
--data_dir ../DATA/${dataset}/imgs/ --save_dir ./${output_dir}  \
--b 4 --dataset ${dataset} --gpu ${gpu} \
--fold ${fold} --tr_size ${tr_size}  --model_type ${model_type} --num_classes 4

tr_size=30
output_dir=sam_${model_type}_seg_${dataset}_f${fold}_tr_${tr_size}_dec
python main_dist_seg.py --src_dir ../DATA/${dataset} \
--data_dir ../DATA/${dataset}/imgs/ --save_dir ./${output_dir}  \
--b 4 --dataset ${dataset} --gpu ${gpu} \
--fold ${fold} --tr_size ${tr_size}  --model_type ${model_type} --num_classes 4

