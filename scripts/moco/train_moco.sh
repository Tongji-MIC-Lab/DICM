#! /usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1 python main_moco.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  /data3/datasets/ILSVRC12
