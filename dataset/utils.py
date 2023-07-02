import os
import pickle
from dataset.Synapse import SynapseDataset
from dataset.ACDC import AcdcDataset
from dataset.SliceLoader import SliceDataset
import torch


def generate_dataset(args):
    split_dir = os.path.join(args.src_dir, "splits.pkl")
    with open(split_dir, "rb") as f:
        splits = pickle.load(f)
    tr_keys = splits[args.fold]['train']
    val_keys = splits[args.fold]['val']
    test_keys = splits[args.fold]['test']

    if args.tr_size < len(tr_keys):
        tr_keys = tr_keys[0:args.tr_size]

    print(tr_keys)
    print(val_keys)
    print(test_keys)


    if args.dataset == 'acdc' or args.dataset == 'ACDC':
        args.img_size = 224
        train_ds = AcdcDataset(keys=tr_keys, mode='train', args=args)
        val_ds = AcdcDataset(keys=val_keys, mode='val', args=args)
        test_ds = AcdcDataset(keys=test_keys, mode='val', args=args)
    else:
        raise NotImplementedError("dataset is not supported:", args.dataset)

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
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=(test_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=test_sampler, drop_last=False
    )

    return train_loader, train_sampler, val_loader, val_sampler, test_loader, test_sampler


def generate_test_loader(key, args):
    key = [key]
    if args.dataset == 'acdc' or args.dataset == 'ACDC':
        args.img_size = 224
        test_ds = AcdcDataset(keys=key, mode='val', args=args)
    else:
        raise NotImplementedError("dataset is not supported:", args.dataset)

    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_ds)
    else:
        test_sampler = None

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=test_sampler, drop_last=False
    )

    return test_loader


def generate_contrast_dataset(args):
    split_dir = os.path.join(args.src_dir, "splits.pkl")
    with open(split_dir, "rb") as f:
        splits = pickle.load(f)
    tr_keys = splits[args.fold]['train']
    val_keys = splits[args.fold]['val']

    if args.tr_size < len(tr_keys):
        tr_keys = tr_keys[0:args.tr_size]

    print(tr_keys)
    print(val_keys)

    if args.dataset == 'acdc' or args.dataset == 'ACDC':
        args.img_size = 224
        train_ds = AcdcDataset(keys=tr_keys, mode='contrast', args=args)
        val_ds = AcdcDataset(keys=val_keys, mode='contrast', args=args)
    else:
        raise NotImplementedError("dataset is not supported:", args.dataset)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False
    )

    return train_loader, val_loader
