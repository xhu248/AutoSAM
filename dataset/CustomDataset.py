import numpy as np
from PIL import Image
import os

from torch.utils.data.dataset import Dataset
from torchvision.transforms.functional import resize
from torchvision import transforms

join = os.path.join


class CustomDataset(Dataset):
    def __init__(self, args, num_files=1000):
        super().__init__()

        file_list = os.listdir(args.data_dir)
        self.files = []

        for k in range(num_files):
            self.files.append(join(args.data_dir, file_list[k]))

        print(f'dataset length: {len(self.files)}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = Image.open(self.files[index])
        img = img.convert(mode='RGB')

        label = img

        img, label = self.prepare_data(img, label)

        return img, label

    def prepare_data(self, img, label):
        aug = transforms.Compose([
            # transforms.RandomResizedCrop(img_size),
            transforms.Resize([1024, 1024]),
            transforms.ToTensor()])

        img = aug(img)
        label = img

        return img, label
