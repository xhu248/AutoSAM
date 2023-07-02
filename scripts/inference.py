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
import nibabel as nib

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
from segment_anything import sam_model_registry, SamPredictor

from models import sam_seg_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from dataset import SynapseDataset, CustomDataset, AcdcDataset, generate_test_loader


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')


parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float,
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
parser.add_argument("--dataset", type=str, default=None)
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

    #model = sam_seg_model_registry[args.model_type](checkpoint=model_checkpoint)
    model = sam_model_registry[args.model_type](checkpoint=model_checkpoint)
    model = model.cuda(args.gpu)

    split_dir = os.path.join(args.src_dir, "splits.pkl")
    with open(split_dir, "rb") as f:
        splits = pickle.load(f)

    tr_keys = splits[args.fold]['train'][0:args.tr_size]
    dataset = AcdcDataset(keys=tr_keys, mode='val', args=args)
    # dataset = CustomDataset(args=args)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    test_sam(model, args)


def infer_feature(data_loader, model, args):

    model.eval()
    embeddings = []
    with torch.no_grad():
        for i, tup in enumerate(data_loader):
            if args.gpu is not None:
                img = tup[0].float().cuda(args.gpu, non_blocking=True)
            else:
                img = tup[0].float()

            print(img.shape, img.max())
            # compute output
            emb = model.get_embedding(img)
            emb = emb.cpu().numpy()
            if len(emb.shape) < 2:
                embeddings.append(emb)
            else:
                for i in range(emb.shape[0]):
                    embeddings.append(emb[i])

    embeddings = np.concatenate(embeddings)
    print(embeddings.shape)
    np.save(os.path.join(args.save_dir, "emb.npy"), embeddings)
    print("embedding saved at:", args.save_dir)


def test_sam(model, args):
    print('Test')
    join = os.path.join
    if not os.path.exists(join(args.save_dir, "infer")):
        os.mkdir(join(args.save_dir, "infer"))
    if not os.path.exists(join(args.save_dir, "label")):
        os.mkdir(join(args.save_dir, "label"))
    if not os.path.exists(join(args.save_dir, "img")):
        os.mkdir(join(args.save_dir, "img"))

    split_dir = os.path.join(args.src_dir, "splits.pkl")
    with open(split_dir, "rb") as f:
        splits = pickle.load(f)
    test_keys = splits[args.fold]['test']

    for key in test_keys:
        preds = []
        labels = []
        imgs = []
        data_loader = generate_test_loader(key, args)
        with torch.no_grad():
            for i, tup in enumerate(data_loader):
                if args.gpu is not None:
                    img = tup[0].float().cuda(args.gpu, non_blocking=True)
                    label = tup[1].long().cuda(args.gpu, non_blocking=True)
                else:
                    img = tup[0]
                    label = tup[1]

                mask = torch.zeros(label.shape)
                for k in range(mask.shape[0]):
                    out = get_mask(img[k], label[k], args.num_classes, model)
                    mask = torch.argmax(out, dim=0)

                preds.append(mask.unsqueeze(0).numpy())
                labels.append(label.cpu().numpy())
                imgs.append(img.cpu().numpy())

            imgs = np.concatenate(imgs, axis=0).squeeze()
            preds = np.concatenate(preds, axis=0).squeeze()
            labels = np.concatenate(labels, axis=0).squeeze()

            print(preds.shape, labels.shape)
            if "." in key:
                key = key.split(".")[0]
            ni_img = nib.Nifti1Image(imgs, affine=np.eye(4))
            ni_pred = nib.Nifti1Image(preds.astype(np.int8), affine=np.eye(4))
            ni_lb = nib.Nifti1Image(labels.astype(np.int8), affine=np.eye(4))
            nib.save(ni_img, join(args.save_dir, 'img', key + '.nii'))
            nib.save(ni_pred, join(args.save_dir, 'infer', key + '.nii'))
            nib.save(ni_lb, join(args.save_dir, 'label', key + '.nii'))
        print("finish saving file:", key)


def get_mask(img, label, num_classes, model):
    resize_transform = ResizeLongestSide(model.image_encoder.img_size)
    original_size = img.shape[1:]
    # generate box for each class
    img = F.interpolate(
            img.unsqueeze(0),
            (model.image_encoder.img_size, model.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
    mask = torch.zeros((num_classes, label.shape[1], label.shape[2]))

    boxes = []
    mask_index = []
    for i in range(1, num_classes):
        lb = label.clone()
        lb[lb != i] = 0
        if torch.sum(lb) == 0: continue
        mask_index.append(i)
        boxes.append(get_box(lb))

    boxes = torch.tensor(boxes, device=model.device)
    print(boxes)
    if boxes.shape[0] != 0:
        batched_input = [
                    {
                        'image': img.squeeze(),
                        'boxes': resize_transform.apply_boxes_torch(boxes, original_size),
                        'original_size': original_size
                    }]
        batched_output = model(batched_input, multimask_output=False)

        for (k, m) in zip(mask_index, batched_output[0]['masks']):
            mask[k] = m
            mask[0] -= mask[k]

        # if torch.sum(lb) == 0:
        #     m = torch.zeros(label.shape)
        # else:
        #     box = torch.tensor(get_box(lb), device=model.device)
        #     batched_input = [
        #         {
        #             'image': img.squeeze(),
        #             'boxes': resize_transform.apply_boxes_torch(box, original_size),
        #             'original_size': original_size
        #         }]
        #     batched_output = model(batched_input, multimask_output=False)
        #     m = batched_output[0]['masks'][0] * batched_output[0]['iou_predictions'][0]

    bg = mask[0]
    bg[bg == 0] = 1
    bg[bg < 0] = 0
    mask[0] = bg

    return mask


def get_box(label):
    """
    given pixel-wise annotation, return box: 1 stands for target organ, 0 stands for background
    """
    img = label.squeeze()
    # get non-background mask
    x1, x2 = 0, img.shape[0]
    y1, y2 = 0, img.shape[1]
    for i in range(1, img.shape[0]):
        if img[i - 1].max() == 0 and img[i].max() > 0:
            x1 = i
            break

    for i in range(img.shape[0] - 2, 0, -1):
        if img[i + 1].max() == 0 and img[i].max() > 0:
            x2 = i
            break

    for j in range(1, img.shape[1]):
        if img[:, j - 1].max() == 0 and img[:, j].max() > 0:
            y1 = j
            break

    for j in range(img.shape[1] - 2, 0, -1):
        if img[:, j + 1].max() == 0 and img[:, j].max() > 0:
            y2 = j
            break
    x1, x2 = x1 - 1, x2 + 1
    y1, y2 = y1 - 1, y2 + 1

    assert x1 < x2, "probem with x dimension when finding non-background, x1:%d, x2:%d" % (x1, x2)
    assert y1 < y2, "probem with y dimension when finding non-background, y1:%d, y2:%d" % (y1, y2)

    return y1, x1, y2, x2


if __name__ == '__main__':
    main()
    # python main_moco.py --data_dir ./data/mmwhs/ --do_contrast --dist-url 'tcp://localhost:10001'
    # --multiprocessing-distributed --world-size 1 --rank 0
