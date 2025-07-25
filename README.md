# DICM
Pytorch code of our recent work "Towards Learned Image Compression for Multiple Intelligent Semantic Analysis Tasks."

# Overview
The overall framework of DICM![overview](https://github.com/Tongji-MIC-Lab/DICM/tree/main/overview.jpg)


# Abstract
Deep neural network (DNN)-based image compression methods have demonstrated superior rate-distortion performance compared to traditional codecs in recent years. However, most existing DNN-based compression methods only optimize signal fidelity at certain bitrate for human perception, neglecting to preserve the richness of semantics in compressed bitstream. This limitation renders the images compressed by existing deep codecs unsuitable for machine vision applications. To bridge the gap between image compression and multiple semantic analysis tasks, an integration of self-supervised learning~(SSL) with deep image compression is proposed in this work to learn generic compressed representations, allowing multiple computer vision tasks to perform semantic analysis from the compressed domain. Specifically, the semantic-guided SSL under bitrate constraint is designed to preserve the semantics of generic visual features and remove the redundancy irrelevant to semantic analysis. Meanwhile, a compression network with high-order spatial interactions is proposed to capture long-range dependencies with low complexity to remove global redundancy. Without incurring decoding cost of pixel-level reconstruction, the features compressed by the proposed method can serve multiple semantic analysis tasks in a compact manner. The experimental results from multiple semantic analysis tasks confirm that the proposed method significantly outperforms traditional codecs and recent deep image compression methods in terms of various analysis performances at similar bitrates.

# Requirements
   Ubuntu 18.04
   Pytorch 1.8.0
   Python 3.8.0
   cuda11.1 + cuDNN v8.0.4 

# Execution order of training
1. You need to use repository <a href="https://github.com/xyupeng/ContrastiveCrop">ContrastiveCrop</a> to train enhanced MoCo for 800 epochs to obtain the semantic-guided self-supervised learning model (if the training epoch is set to 200, it will degrade the performance of compressed features on machine vision tasks). Execute the command as follows
   python DDP_moco_ccrop.py configs/IN1K/mocov2_ccrop.py 

2. The parameters of the pre-trained self-supervised learning model are frozen, and the parameters of the redundant filtering module are being trained. Execute the command as follows
scripts/train_omni_fea/pretrain_IF.sh 

3. The parameters of the self-supervised learning model and the redundant filtering module are trained jointly. Execute the command as follows
scripts/train_omni_fea/train_omni.sh 

4. All parameters except those of the compressed network are frozen, and the following command is used to train the compressed network
scripts/train_omni_fea/train_fea_codec.sh 

# Acknowledgments
Codebase from <a href="https://github.com/xyupeng/ContrastiveCrop">ContrastiveCrop</a>, <a href="https://github.com/damo-cv/entroformer">Entroformer</a> and <a href="https://arxiv.org/abs/2207.01932">Omni-ICM</a>.
