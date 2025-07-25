#! /usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python main_moco_cf.py \
  -a resnet50 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10014' --multiprocessing-distributed --world-size 1 --rank 0 \
  --mlp --moco-t 0.2 --aug-plus --cos \
  --save_dir ./ckpt/train_omni_fea/main/contr1_rate0.1_dist0.1_l1_new \
  --optimizer sgd \
  --pretraining_stage \
  --lambda_contr 1 --lambda_rate 0.1 --lambda_dist 0.1 --compress_metric l1 \
  --lr 1e-3 --epochs 200 --epochs_freeze_backbone 0 \
  --resume ckpt/train_omni_fea/main/contr1_rate0.1_dist0.1_l1/checkpoint_0299.pth.tar \
  --pretrained ckpt/train_omni_fea/pretrain_IF/contr0_rate1_dist128_l1/checkpoint_0019.pth.tar \

#  $PATH_OF_YOUR_IMAGENET_DATASET$