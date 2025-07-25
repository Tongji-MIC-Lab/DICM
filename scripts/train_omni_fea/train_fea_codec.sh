#! /usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1 python main_moco_cf.py \
  -a resnet50 \
  --batch-size 32 \
  --dist-url 'tcp://localhost:10012' --multiprocessing-distributed --world-size 1 --rank 0 \
  --mlp --moco-t 0.2 --aug-plus --cos \
  --save_dir ./ckpt/train_fea_codec/ab_fealoss_lamnda08_768_512_256 \
  --optimizer adamw \
  --na bidirectional \
  --scale 2 \
  --num_parameter 2 \
  --lambda_contr 0 --lambda_rate 0.08 --lambda_dist 2048 --compress_metric l2 \
  --feats_distortion --lambda_feats_2 384 --lambda_feats_3 256 --lambda_feats_4 192 --feats_loss_type l2 \
  --lr 1e-4 --epochs 20 \
  --position_num 7 \
  --mask_ratio 0.5 \
  --attn_topk 32 \
  --grad_norm_clip 1.0 \
  --pretrained ckpt/train_omni_fea/main/contr1_rate0.1_dist0.1_l1/checkpoint_0299.pth.tar \
#  $PATH_OF_YOUR_IMAGENET_DATASET$