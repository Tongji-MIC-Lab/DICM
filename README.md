# DICM
Pytorch code of our recent work "Towards Learned Image Compression for Multiple Intelligent Semantic Analysis Tasks."

## Requirements:
   Ubuntu 18.04
   Pytorch 1.8.0
   Python 3.8.0
   cuda11.1 + cuDNN v8.0.4 

## Execution order of training：
1. You need to use repository <a href="https://github.com/xyupeng/ContrastiveCrop">ContrastiveCrop</a> to train enhanced MoCo for 800 epochs to obtain the semantic-guided self-supervised learning model (if the training epoch is set to 200, it will degrade the performance of compressed features on machine vision tasks). Execute the command as follows
   python DDP_moco_ccrop.py configs/IN1K/mocov2_ccrop.py 

2. The parameters of the pre-trained self-supervised learning model are frozen, and the parameters of the redundant filtering module are being trained. Execute the command as follows
scripts/train_omni_fea/pretrain_IF.sh 

3. The parameters of the self-supervised learning model and the redundant filtering module are trained jointly. Execute the command as follows
scripts/train_omni_fea/train_omni.sh 

4. All parameters except those of the compressed network are frozen, and the following command is used to train the compressed network
scripts/train_omni_fea/train_fea_codec.sh 

## Acknowledgments
Codebase from <a href="https://github.com/xyupeng/ContrastiveCrop">ContrastiveCrop</a>, <a href="https://github.com/damo-cv/entroformer">Entroformer</a> and <a href="https://arxiv.org/abs/2207.01932">Omni-ICM</a>.
