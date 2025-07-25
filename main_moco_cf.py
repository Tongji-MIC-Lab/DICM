#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import builtins
import math
import os
import random
import shutil
import time
import warnings
from collections import OrderedDict
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from dataset import build_dataset, build_dataset_ccrop
import moco.builder
from moco import resnet
from utils import build_logger, format_time
from get_args import get_parser_args

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


def update_box(eval_train_loader, model, len_ds, logger, args, t=0.05):
    if args.rank==0:
        logger.info(f'==> Start updating boxes...')
    model.eval()
    boxes = []
    t1 = time.time()
    for cur_iter, (images, _) in enumerate(eval_train_loader):  # drop_last=False
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _,_,_,_,_,_,_,feat_map = model.forward_with_feats(images, training=False, use_codec=False)  # (N, C, H, W)
            feat_map=feat_map[3]
        N, Cf, Hf, Wf = feat_map.shape
        eval_train_map = feat_map.sum(1).view(N, -1)  # (N, Hf*Wf)
        eval_train_map = eval_train_map - eval_train_map.min(1, keepdim=True)[0]
        eval_train_map = eval_train_map / eval_train_map.max(1, keepdim=True)[0]
        eval_train_map = eval_train_map.view(N, 1, Hf, Wf)
        eval_train_map = F.interpolate(eval_train_map, size=images.shape[-2:], mode='bilinear')  # (N, 1, Hi, Wi)
        Hi, Wi = images.shape[-2:]

        for hmap in eval_train_map:
            hmap = hmap.squeeze(0)  # (Hi, Wi)

            h_filter = (hmap.max(1)[0] > t).int()
            w_filter = (hmap.max(0)[0] > t).int()

            h_min, h_max = torch.nonzero(h_filter).view(-1)[[0, -1]] / Hi  # [h_min, h_max]; 0 <= h <= 1
            w_min, w_max = torch.nonzero(w_filter).view(-1)[[0, -1]] / Wi  # [w_min, w_max]; 0 <= w <= 1
            boxes.append(torch.tensor([h_min, w_min, h_max, w_max]))

    boxes = torch.stack(boxes, dim=0).cuda()  # (num_iters, 4)
    gather_boxes = [torch.zeros_like(boxes) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_boxes, boxes)
    all_boxes = torch.stack(gather_boxes, dim=1).view(-1, 4)
    all_boxes = all_boxes[:len_ds]
    if logger is not None:  # cfg.rank == 0
        t2 = time.time()
        epoch_time = format_time(t2 - t1)
        logger.info(f'Update box: {epoch_time}')
    return all_boxes

def main():
    args = get_parser_args()
    content = str(args).replace(', ', ',\n\t') + '\n'
    print(content)
    args.mean=mean
    args.std=std

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # build logger, writer
    logger, writer = None, None
    if args.rank==0:
        writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'tensorboard'))
        logger = build_logger(args.save_dir, 'pretrain')

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = moco.builder.MoCoComFea(
        resnet.resnet50_cf,args,
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
    print(model)

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")
            if args.pretraining_IF:
                state_dict = checkpoint['moco_state']
            else:
                state_dict = checkpoint['state_dict']   # 'moco_state'

            # remove `module.`
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            state_dict = new_state_dict

            # remove 'encoder_q.'
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[10:]
                new_state_dict[name] = v
            state_dict = new_state_dict

            print('load encoder_q...')
            msg_q = model.encoder_q.load_state_dict(state_dict, strict=False)
            print('msg_q: {}'.format(msg_q))

            print('load encoder_k...')
            msg_k = model.encoder_k.load_state_dict(state_dict, strict=False)
            print('msg_k: {}'.format(msg_k))

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(
                (args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, find_unused_parameters=True, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(
                model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print('optimizer is : {}'.format(args.optimizer))

    ## dataset
    args.aug_resize = 224 if args.pretraining_stage else 256  # hyper need the image to be times of 128

    train_dataset = build_dataset_ccrop(args)
    len_ds = len(train_dataset)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(
                train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    if args.pretraining_stage:
        eval_train_set = build_dataset(args)
        if args.distributed:
            eval_train_sampler = torch.utils.data.distributed.DistributedSampler(eval_train_set, shuffle=False)
        else:
            eval_train_sampler = None
        eval_train_loader = torch.utils.data.DataLoader(
            eval_train_set,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            sampler=eval_train_sampler,
            drop_last=False
        )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.pretraining_stage:
                train_dataset.boxes = checkpoint['boxes'].cpu()
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # start ContrastiveCrop
        if args.pretraining_stage:
            train_dataset.use_box = epoch >= args.warmup_epochs

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, logger, writer, args)

        # update boxes; all processes
        if args.pretraining_stage:
            if epoch != args.epochs and epoch % args.loc_interval == 0 and epoch >= 20:
                # all_boxes: tensor (len_ds, 4); (h_min, w_min, h_max, w_max)
                all_boxes = update_box(eval_train_loader, model.module.encoder_q, len_ds, logger,
                                       args, t=args.box_thresh)  # on_cuda=True
                assert len(all_boxes) == len_ds
                train_dataset.boxes = all_boxes.cpu()

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            if (epoch+1) % args.save_interval == 0:
                save_checkpoint(args, {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'boxes': train_dataset.boxes,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch))


def train(train_loader, model, criterion, optimizer, epoch, logger, writer, args):
    if args.pretraining_stage:
        # freeze the backbone in first N epochs to pretrain the IF module
        if epoch < args.epochs_freeze_backbone:
            for p in model.module.parameters():
                p.requires_grad = False
            for p in model.module.encoder_q.IF_enc.parameters():
                p.requires_grad = True
            for p in model.module.encoder_q.IF_dec.parameters():
                p.requires_grad = True
            for p in model.module.encoder_q.IF_FM.parameters():  # use FactorizedEntropy
                p.requires_grad = True
        else:
            for p in model.module.parameters():
                p.requires_grad = True
    else:
        # for the codec tranining phase, fix all the parameters except codec itself
        for p in model.module.parameters():
            p.requires_grad = False
        for p in model.module.encoder_q.codec_enc.parameters():
            p.requires_grad = True
        for p in model.module.encoder_q.codec_dec.parameters():
            p.requires_grad = True
        for p in model.module.encoder_q.hyper_enc.parameters():
            p.requires_grad = True
        for p in model.module.encoder_q.hyper_dec.parameters():
            p.requires_grad = True
        for p in model.module.encoder_q.cit_ar.parameters():
            p.requires_grad = True
        for p in model.module.encoder_q.cit_pn.parameters():
            p.requires_grad = True
        for p in model.module.encoder_q.prob_model.parameters():
            p.requires_grad = True

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses_dist = AverageMeter('Dist', ':.4f')
    losses_dist_feats = AverageMeter('Dist_Feats', ':.4f')
    losses_rate = AverageMeter('Rate(all)', ':.4f')
    losses_rate_y = AverageMeter('Rate(y)', ':.4f')
    losses_rate_z = AverageMeter('Rate(z)', ':.4f')
    losses_contr = AverageMeter('Loss_contr', ':.4f')
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses_dist, losses_dist_feats, losses_rate,
            losses_rate_y, losses_rate_z, losses_contr, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    num_iter=len(train_loader)
    # switch to train mode
    model.train()

    # define the compress metric
    if args.compress_metric == 'l1':
        com_loss = nn.L1Loss()
    elif args.compress_metric == 'l2':
        com_loss = nn.MSELoss()
    elif args.compress_metric == 'smooth_l1':
        com_loss = nn.SmoothL1Loss()
    else:
        raise NotImplementedError

    if args.feats_loss_type == 'l1':
        feats_dist_func = nn.L1Loss()
    elif args.feats_loss_type == 'l2':
        feats_dist_func = nn.MSELoss()
    elif args.feats_loss_type == 'smooth_l1':
        feats_dist_func = nn.SmoothL1Loss()
    else:
        raise NotImplementedError

    end = time.time()
    time1=time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        if args.pretraining_stage:
            output, target, fea, omni_fea, IF_y_prob, omni_fea_hat, comp_y_prob, comp_z_prob, feats = model(
                im_q=images[0], im_k=images[1], use_codec=False)
        else:
            output, target, fea, omni_fea, IF_y_prob, omni_fea_hat, comp_y_prob, comp_z_prob, feats = model(
                im_q=images[0], im_k=images[1], use_codec=True)

        # distortion loss
        if args.pretraining_stage:
            distortion = com_loss(fea, omni_fea)
        else:
            distortion = com_loss(omni_fea, omni_fea_hat)

        # feature-level loss
        if args.pretraining_stage:
            distortion_feats = torch.tensor(0)
        else:
            if args.feats_distortion:
                with torch.no_grad():
                    feats_gt = model.module.encoder_q.forward_with_feats(
                        images[0], training=False, use_codec=False)[-1]
                distortion_feats = feature_distortion(
                    args, feats, feats_gt, feats_dist_func)
            else:
                distortion_feats = torch.tensor(0)

        # rate loss
        b, c, h, w = images[0].shape
        num_pixels = b*h*w
        if args.pretraining_stage:
            bits_fea_y = -torch.sum(torch.log2(IF_y_prob))
            bits_fea_z = torch.tensor(0)
        else:
            bits_fea_y=comp_y_prob.sum() / np.log(2)
            bits_fea_z = -torch.sum(torch.log2(comp_z_prob))
        rate_fea_y = bits_fea_y / num_pixels
        rate_fea_z = bits_fea_z / num_pixels
        rate_fea = rate_fea_y + rate_fea_z

        # contrastive loss
        loss_contr = criterion(output, target)

        # total loss
        loss = args.lambda_dist * distortion + args.lambda_rate * \
            rate_fea + args.lambda_contr * loss_contr + distortion_feats

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses_dist.update(distortion.item(), images[0].size(0))
        losses_dist_feats.update(distortion_feats.item(), images[0].size(0))
        losses_rate.update(rate_fea.item(), images[0].size(0))
        losses_rate_y.update(rate_fea_y.item(), images[0].size(0))
        losses_rate_z.update(rate_fea_z.item(), images[0].size(0))
        losses_contr.update(loss_contr.item(), images[0].size(0))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # print info on screen
        if i % args.print_freq == 0:
            progress.display(i)
        #print info in log
        if (i + 1) % 10*args.print_freq == 0 and args.rank==0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(f'Epoch [{epoch}][{i + 1}/{num_iter}] - '
                        f'data_time: {data_time.avg:.3f},     '
                        f'batch_time: {batch_time.avg:.3f},     '
                        f'lr: {lr:.5f},     '
                        f'loss: {loss:.3f}({losses.avg:.3f})')

    epoch_time=format_time(time.time()-time1)
    print('This epoch_{} consumes time: {}' .format(epoch, epoch_time))
    if args.rank==0:
        logger.info(f'Epoch [{epoch}] - epoch_time: {epoch_time}, '
                    f'train_loss: {losses.avg:.3f}')

    # tensorboard
    if args.rank==0:
        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Pretrain/lr', lr, epoch)
        writer.add_scalar('Pretrain/loss', losses.avg, epoch)



def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    os.makedirs(args.save_dir, exist_ok=True)
    filename = os.path.join(args.save_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(
            args.save_dir, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous(
            ).view(-1).float().sum(0, keepdim=True)
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def feature_distortion(args, feats, feats_gt, func):
    num_feats = len(feats)
    feat_dist = 0

    feat_dist += args.lambda_feats_2 * func(feats[1], feats_gt[1].detach())
    feat_dist += args.lambda_feats_3 * func(feats[2], feats_gt[2].detach())
    feat_dist += args.lambda_feats_4 * func(feats[3], feats_gt[3].detach())

    return feat_dist


if __name__ == '__main__':
    main()
