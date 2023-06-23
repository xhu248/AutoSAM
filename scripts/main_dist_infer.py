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
from datetime import datetime

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
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as models


from loss_functions.dice_loss import SoftDiceLoss
from loss_functions.metrics import dice_pytorch, SegmentationMetric

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from dataset import SynapseDataset, AcdcDataset


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')


parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

parser.add_argument('--model_type', type=str, default="vit_l", help='path to splits file')
parser.add_argument('--src_dir', type=str, default=None, help='path to splits file')
parser.add_argument('--data_dir', type=str, default=None, help='path to datafolder')
parser.add_argument("--img_size", type=int, default=256)
parser.add_argument("--classes", type=int, default=8)
parser.add_argument("--do_contrast", default=False, action='store_true')
parser.add_argument("--slice_threshold", type=float, default=0.05)
parser.add_argument("--num_classes", type=int, default=8)
parser.add_argument("--fold", type=int, default=0)
parser.add_argument("--tr_size", type=int, default=1)
parser.add_argument("--save_dir", type=str, default=None)
parser.add_argument("--load_saved_model", action='store_true',
                        help='whether freeze encoder of the segmenter')
parser.add_argument("--saved_model_path", type=str, default=None)
parser.add_argument("--load_pseudo_label", default=False, action='store_true')


def main():
    args = parser.parse_args()

    args.distributed = False

    if args.model_type=='vit_h':
        model_checkpoint = 'sam_vit_h_4b8939.pth'
    elif args.model_type == 'vit_l':
        model_checkpoint = 'sam_vit_l_0b3195.pth'
    elif args.model_type == 'vit_b':
        model_checkpoint = 'sam_vit_b_01ec64.pth'

    model = sam_model_registry[args.model_type](checkpoint=model_checkpoint)
    model = model.cuda(args.gpu)
    predictor = SamPredictor(model)

    split_dir = os.path.join(args.src_dir, "splits.pkl")
    with open(split_dir, "rb") as f:
        splits = pickle.load(f)

    tr_keys = splits[args.fold]['train'][0:args.tr_size]
    dataset = AcdcDataset(keys=tr_keys, mode='val', args=args)
    # dataset = CustomDataset(args=args)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    infer(data_loader, predictor, args)


def infer(data_loader, predictor, args):
    print('Test')
    metric_val = SegmentationMetric(args.num_classes)
    metric_val.reset()

    with torch.no_grad():
        for i, tup in enumerate(data_loader):
            # measure data loading time

            if args.gpu is not None:
                img = tup[0].float().cuda(args.gpu, non_blocking=True)
                label = tup[1].long().cuda(args.gpu, non_blocking=True)
            else:
                img = tup[0]
                label = tup[1]
            scribble = tup[2]

            b, c, h, w = img.shape
            predictor.set_image(img.squeeze().view(h, w, 3))
            # compute output
            masks, scores, logits = predictor.predict(
                point_coords=np.asarray([[10, 10], [20, 20], [21, 22]]),
                point_labels=np.asarray([1, 1, 0]),
                multimask_output=True,
            )
            exit()

            if i % args.print_freq == 0:
                print("Index:%f, mean Dice:%.4f" % (i, Dice))

    _, _, Dice = metric_val.get()
    print("Overall mean dice score is:", Dice)
    print("Finished test")


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
