#! /usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python main_moco_cf.py \
  -a resnet50 \
  --batch-size 64 \
  --dist-url 'tcp://localhost:10003' --multiprocessing-distributed --world-size 1 --rank 0 \
  --mlp --moco-t 0.2 --aug-plus --cos \
  --save_dir ./ckpt/train_omni_fea/pretrain_IF/contr0_rate1_dist128_l1_new \
  --optimizer adam \
  --save_interval 1 \
  --pretraining_stage \
  --pretraining_IF \
  --lambda_contr 0 --lambda_rate 1 --lambda_dist 128 --compress_metric l1 \
  --lr 1e-4 --epochs 20 --epochs_freeze_backbone 20 \
  --pretrained ckpt/pretrained/moco_ccrop_v2_800ep_pretrain.pth
