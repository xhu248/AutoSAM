import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models

import networks.moco as moco
from datasets.slice_loader.mmwhs import MMWHS
from loss_functions.dice_loss import SoftDiceLoss
from loss_functions.metrics import dice_pytorch, SegmentationMetric

from networks.unet_con import GlobalConUnet, MLP, SupConUnet

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
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
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
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
parser.add_argument('--moco-k', default=192, type=int,
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

parser.add_argument('--data_dir', type=str, default=None, help='path to custom dataset')
parser.add_argument("--patch_size", type=tuple, default=(512, 512))
parser.add_argument("--classes", type=int, default=8)
parser.add_argument("--do_contrast", default=True, action='store_true')
parser.add_argument("--slice_threshold", type=float, default=0.05)
parser.add_argument("--num_classes", type=int, default=8)



def main():
    args = parser.parse_args()

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
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
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
    # create model

    print("=> creating model '{}'".format(args.arch))
    model = SupConUnet(num_classes=args.num_classes)

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
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
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

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code

    train_loader, train_sampler, val_loader, val_sampler, test_loader, test_sampler = generate_dataset(args)

    save_dir = "save/moco/mmwhs"
    filename = os.path.join(save_dir, 'checkpoint_b%d_k%d.pth.tar' % (args.batch_size, args.moco_k))
    writer = SummaryWriter(os.path.join(save_dir, 'tensorboard'))

    best_loss = 100
    for epoch in range(args.start_epoch, args.epochs):
        is_best = False
        if args.distributed:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, optimizer, epoch, args, writer)
        loss = validate(val_loader, model, epoch, args, writer)

        if loss < best_loss:
            is_best = True
            best_loss = loss

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=is_best, filename=filename)

    test(test_loader, model, args)


def train(train_loader, model, optimizer, epoch, args, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    dice_loss = SoftDiceLoss(batch_dice=True, do_bg=False)
    ce_loss = torch.nn.CrossEntropyLoss()

    # switch to train mode
    model.train()

    end = time.time()
    for i, tup in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            img = tup[0].float().cuda(args.gpu, non_blocking=True)
            label = tup[1].float().cuda(args.gpu, non_blocking=True)
        else:
            img = tup[0]
            label = tup[1]


        # compute output
        pred = model(img)
        pred_softmax = F.softmax(pred, dim=1)
        loss = ce_loss(pred, label.squeeze()) + dice_loss(pred_softmax, label.squeeze())

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        writer.add_scalar('train_loss', loss, global_step=i + epoch * len(train_loader))

        if i % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'loss {loss:.4f}'.format(epoch, i, len(train_loader), loss=loss.item()))


def validate(val_loader, model, epoch, args, writer):
    print('VALIDATE')
    loss_list = []
    dice_list = []
    dice_loss = SoftDiceLoss(batch_dice=True, do_bg=False)
    model.eval()

    with torch.no_grad():
        for i, tup in enumerate(val_loader):
            # measure data loading time

            if args.gpu is not None:
                img = tup[0].float().cuda(args.gpu, non_blocking=True)
                label = tup[1].float().cuda(args.gpu, non_blocking=True)
            else:
                img = tup[0]
                label = tup[1]

            # compute output
            pred = model(img)
            pred_softmax = F.softmax(pred, dim=1)

            loss = dice_loss(pred_softmax, label.squeeze())  # self.ce_loss(pred, target.squeeze())
            loss_list.append(loss.item())

    print('Epoch: %2d Loss: %.4f' % (epoch, np.mean(loss_list)))
    writer.add_scalar("val_loss", np.mean(loss_list), epoch)
    return np.mean(loss_list)


def test(data_loader, model, args):
    print('Test')
    metric_val = SegmentationMetric(args.num_classes)
    metric_val.reset()
    model.eval()

    with torch.no_grad():
        for i, tup in enumerate(data_loader):
            # measure data loading time

            if args.gpu is not None:
                img = tup[0].float().cuda(args.gpu, non_blocking=True)
                label = tup[1].float().cuda(args.gpu, non_blocking=True)
            else:
                img = tup[0]
                label = tup[1]

            # compute output
            pred = model(img)
            pred_softmax = F.softmax(pred, dim=1)

            metric_val.update(label.squeeze(dim=1), pred_softmax)
            pixAcc, mIoU, Dice = metric_val.get()

            if i % args.print_freq == 0:
                print("Index:%f, mean Dice:%.4f" % (i, Dice))

    _, _, Dice = metric_val.get()
    print("Overall mean dice score is:", Dice)
    print("Finished test")


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    # torch.save(state, filename)
    if is_best:
        torch.save(state, filename)
        shutil.copyfile(filename, 'model_best.pth.tar')


def generate_dataset(args):
    split_dir = os.path.join(args.data_dir, "splits.pkl")
    with open(split_dir, "rb") as f:
        splits = pickle.load(f)
    tr_keys = splits[0]['train']
    val_keys = splits[0]['val']
    test_keys = splits[0]['test']

    train_ds = MMWHS(keys=tr_keys, purpose='train', args=args)
    val_ds = MMWHS(keys=val_keys, purpose='val', args=args)
    test_ds = MMWHS(keys=test_keys, purpose='val', args=args)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_ds)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=(test_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=test_sampler, drop_last=True
    )

    return train_loader, train_sampler, val_loader, val_sampler, test_loader, test_sampler


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
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()

    # python main_moco.py --data_dir ./data/mmwhs/ --do_contrast --dist-url 'tcp://localhost:10001'
    # --multiprocessing-distributed --world-size 1 --rank 0
