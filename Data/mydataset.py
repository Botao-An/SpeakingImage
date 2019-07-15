import os
import numpy as np
import torch
import torchvision
from torchvision import transforms
import torch.utils.data as data
from . import utils

class MyDataset(data.Dataset):

    train_cover_folder = 'CoverData/train_cover'
    val_cover_folder = 'CoverData/val_cover'
    test_cover_folder = 'CoverData/test_cover'

    train_secret_folder = 'SecretData/train_secret'
    val_secret_folder = 'SecretData/val_secret'
    test_secret_folder = 'SecretData/test_secret'

    img_extension = '.jpg'
    csv_extension = '.csv'

    def __init__(self,
                 root_dir,
                 mode='train',
                 cover_transform=None,
                 secret_transform = None,
                 cover_loader=utils.pil_loader,
                 secret_loader = utils.csv_loader):
        self.root_dir = root_dir
        self.mode = mode
        self.cover_transform = cover_transform
        self.secret_transform = secret_transform
        self.cover_loader = cover_loader
        self.secret_loader = secret_loader

        if self.mode.lower() == 'train':
            self.train_cover_data = utils.get_files(os.path.join(root_dir, self.train_cover_folder))
            self.train_secret_data = utils.get_files(os.path.join(root_dir, self.train_secret_folder))

        elif self.mode.lower() == 'val':
            self.val_cover_data = utils.get_files(os.path.join(root_dir, self.val_cover_folder))
            self.val_secret_data = utils.get_files(os.path.join(root_dir, self.val_secret_folder))

        elif self.mode.lower() == 'test':
            self.test_cover_data = utils.get_files(os.path.join(root_dir, self.test_cover_folder))
            self.test_secret_data = utils.get_files(os.path.join(root_dir, self.test_secret_folder))

        else:
            raise RuntimeError("Unexpected dataset mode. Supported modes are: train, val and test")

    def __getitem__(self, index):

        if self.mode.lower() == 'train':
            cover_path, secret_path = self.train_cover_data[index], self.train_secret_data[index]

        elif self.mode.lower() == 'val':
            cover_path, secret_path = self.val_cover_data[index], self.val_secret_data[index]

        elif self.mode.lower() == 'test':
            cover_path, secret_path = self.test_cover_data[index], self.test_secret_data[index]

        else:
            raise RuntimeError("Unexpected dataset mode. Supported modes are: train, val and test")

        cover = self.cover_loader(cover_path)
        secret = self.secret_loader(secret_path)

        if self.cover_transform is not None:
            cover = self.cover_transform(cover)

        if self.secret_transform is not None:
            secret = self.secret_transform(secret)

        return cover, secret

    def __len__(self):

        if self.mode.lower() == 'train':
            return len(self.train_cover_data)
        elif self.mode.lower() == 'val':
            return len(self.val_cover_data)
        elif self.mode.lower() == 'test':
            return len(self.test_cover_data)
        else:
            raise RuntimeError("Unexpected dataset mode. Supported modes are: train, val and test")