import argparse
import torchvision.models as models


def get_parser_args():
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('-data',  default='/root/data1/tzs/datasets/Imagenet/',  #/root/tzs/datasets/Imagenet/
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://localhost:10015', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    # moco specific configs:
    parser.add_argument('--moco-dim', default=128, type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument('--moco-k', default=65536, type=int,
                        help='queue size; number of negative keys (default: 65536)')
    parser.add_argument('--moco-m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco-t', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')

    # options for moco v2
    parser.add_argument('--mlp', action='store_true',
                        help='use mlp head')
    parser.add_argument('--aug-plus', action='store_true',
                        help='use moco v2 data augmentation')
    parser.add_argument('--cos', action='store_true',
                        help='use cosine lr schedule')

    # options for crop
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--box_thresh', type=float, default=0.1)

    # other setting
    parser.add_argument('--save_dir', default=0, type=str, metavar='', help='')
    parser.add_argument("--compress_metric", type=str, default='l2',
                        choices=['l1', 'l2', 'smooth_l1'])
    parser.add_argument("--lambda_contr", type=float, default=1)
    parser.add_argument("--lambda_dist", type=float, default=256)
    parser.add_argument("--lambda_rate", type=float, default=0.1)
    parser.add_argument("--epochs_freeze_backbone", type=int, default=2)
    parser.add_argument('--normalize_dist', action='store_true')
    parser.add_argument('--pretrained', default='path to pretrained model')
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--loc_interval', type=int, default=20)

    # Configure for Transfomer Entropy Model
    parser.add_argument("--no_rpe_shared", dest="rpe_shared", action="store_false", default=True, help="Position Shared in layers.")
    parser.add_argument("--mask_ratio", type=float, default=0., help="Pretrain model: mask ratio.")
    parser.add_argument("--dim_embed", type=int, default=384, help="Dimension of transformer embedding.")
    parser.add_argument("--depth", type=int, default=6, help="Depth of CiT.")
    parser.add_argument("--K", type=int, default=1, help="the number of Mix Hyperprior.")
    parser.add_argument("--heads", type=int, default=6, help="Number of transformer head.")
    parser.add_argument("--mlp_ratio", type=int, default=4, help="Ratio of transformer MLP.")
    parser.add_argument("--dim_head", type=int, default=64, help="Dimension of transformer head.")
    parser.add_argument("--trans_no_norm", dest="trans_norm", action="store_false", default=True, help="Use LN in transformer.")
    parser.add_argument("--dropout", type=float, default=0., help="Dropout ratio.")
    parser.add_argument("--position_num", type=int, default=7, help="Position information num.")
    parser.add_argument("--att_noscale", dest="att_scale", action="store_false", default=True, help="Use Scale in Attention.")
    parser.add_argument("--scale", type=int, default=2, help="Downscale of hyperprior of CiT.")
    parser.add_argument("--attn_topk", type=int, default=-1, help="Top K filter for Self-attention.")
    parser.add_argument("--grad_norm_clip", type=float, default=0., help="grad_norm_clip.")
    parser.add_argument("--warmup", type=float, default=0.05, help="Warm up.")
    parser.add_argument("--segment", type=int, default=1, help="Segment for Large Patchsize.")
    parser.add_argument("--na", type=str, default='bidirectional', help="Entropy model for prediction manner.")
    parser.add_argument("--num_parameter", type=int, default=3,
                        help="distribution parameter num: 1 for sigma, 2 for mean&sigma, 3 for mean&sigma&pi")
    parser.add_argument("--table_range", type=int, default=128, help="range of feature")

    # for codec traininng
    parser.add_argument('--pretraining_stage', action='store_true',
                        help='In the pretraining stage, the feature codec would not be used !'
                        'and only information_filter used for information filtering.')
    parser.add_argument('--pretraining_IF', default=False, action='store_true')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--feats_distortion', action='store_true',
                        help='using feature-level loss')
    parser.add_argument('--lambda_feats_2', type=float, default=0,
                        help='different lambda with different layers, '
                        'layer1 is calculated by lambda_dist, so start at layer2')
    parser.add_argument('--lambda_feats_3', type=float, default=0)
    parser.add_argument('--lambda_feats_4', type=float, default=0)
    parser.add_argument('--feats_loss_type', type=str,
                        default='l1', choices=['l1', 'l2', 'smooth_l1'])

    return parser.parse_args()