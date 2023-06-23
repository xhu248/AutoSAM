import pickle
import numpy as np
from PIL import Image
import os

import torch
from torchvision import transforms, utils
from torch.utils.data.dataset import Dataset
from torchvision.transforms.functional import resize

from batchgenerators.utilities.file_and_folder_operations import *
from batchgenerators.transforms.abstract_transforms import Compose, RndTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform, ResizeTransform
from batchgenerators.transforms.color_transforms import BrightnessTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor

join = os.path.join


class AcdcDataset(Dataset):
    def __init__(self, keys, args, mode='train'):
        super().__init__()
        self.patch_size = (args.img_size, args.img_size)
        self.files = []
        self.mode = mode

        for f in os.listdir(args.data_dir):
            if f.split("_frame")[0] in keys:
                slices = subfiles(join(args.data_dir, f))
                for sl in slices:
                    self.files.append(sl)

        print(f'dataset length: {len(self.files)}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = Image.open(self.files[index])
        label = Image.open(self.files[index].replace('imgs/', 'annotations/'))
        label = np.asarray(label)
        # scribble = Image.open(self.files[index].replace('imgs/', 'scribbles/'))
        # scribble = np.asarray(scribble)
        img = np.asarray(img).astype(np.float32).transpose([2, 0, 1])
        img = (img - img.min()) / (img.max() - img.min())
        if self.mode == 'contrast':
            img1, img2 = self.transform_contrast(img)
            return img1, img2
        else:
            img, label = self.transform(img, label)
            return img, label

    def transform_contrast(self, img):
        # the image and label should be [batch, c, x, y, z], this is the adapatation for using batchgenerators :)
        data_dict = {'data': img[None]}
        tr_transforms = [  # CenterCropTransform(crop_size=target_size),
            BrightnessTransform(mu=1, sigma=1, p_per_sample=0.5),
            GammaTransform(p_per_sample=0.5),
            GaussianNoiseTransform(p_per_sample=0.5),
            ResizeTransform(target_size=self.patch_size, order=1),  # resize
            MirrorTransform(axes=(1,)),
            SpatialTransform(patch_size=self.patch_size, random_crop=False,
                             patch_center_dist_from_border=self.patch_size[0] // 2,
                             do_elastic_deform=True, alpha=(100., 350.), sigma=(40., 60.),
                             do_rotation=True, p_rot_per_sample=0.5,
                             angle_x=(-0.1, 0.1), angle_y=(0, 1e-8), angle_z=(0, 1e-8),
                             scale=(0.5, 1.9), p_scale_per_sample=0.5,
                             border_mode_data="nearest", border_mode_seg="nearest"),
        ]

        train_transform = Compose(tr_transforms)
        data_dict = train_transform(**data_dict)
        img1 = data_dict.get('data')[0]
        data_dict = train_transform(**data_dict)
        img2 = data_dict.get('data')[0]
        return img1, img2

    def transform(self, img, label):
        # normalize to [0, 1]
        data_dict = {'data': img[None], 'seg': label[None, None]}
        if self.mode == 'train':
            aug_list = [  # CenterCropTransform(crop_size=target_size),
                BrightnessTransform(mu=1, sigma=1, p_per_sample=0.5),
                GammaTransform(p_per_sample=0.5),
                GaussianNoiseTransform(p_per_sample=0.5),
                ResizeTransform(target_size=self.patch_size, order=1),  # resize
                MirrorTransform(axes=(1,)),
                SpatialTransform(patch_size=self.patch_size, random_crop=False,
                                 patch_center_dist_from_border=self.patch_size[0] // 2,
                                 do_elastic_deform=True, alpha=(100., 350.), sigma=(40., 60.),
                                 do_rotation=True, p_rot_per_sample=0.5,
                                 angle_x=(-0.1, 0.1), angle_y=(0, 1e-8), angle_z=(0, 1e-8),
                                 scale=(0.5, 1.9), p_scale_per_sample=0.5,
                                 border_mode_data="nearest", border_mode_seg="nearest"),
                NumpyToTensor(),
            ]

            aug = Compose(aug_list)
        else:
            aug_list = [
                ResizeTransform(target_size=self.patch_size, order=1),
                NumpyToTensor(),
            ]
            aug = Compose(aug_list)

        data_dict = aug(**data_dict)
        img = data_dict.get('data')[0]
        label = data_dict.get('seg')[0]
        return img, label



