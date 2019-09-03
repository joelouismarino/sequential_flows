import os
import torch
import numpy as np
from torch.utils.data import Dataset


class MovingMNIST(Dataset):
    """
    Dataset class for moving MNIST dataset.

    Args:
        path (str): path to the .npy dataset
        transform (torchvision.transforms): image/video transforms
    """
    def __init__(self, path, transform=None):
        assert os.path.exists(path), 'Invalid path to Moving MNIST data set: ' + path
        self.transform = transform
        self.data = np.load(path)

    def __getitem__(self, ind):
        imgs = self.data[:,ind,:,:].astype('float32')
        s, h, w = imgs.shape
        imgs = imgs.reshape(s, 1, h, w)
        if self.transform is not None:
            # apply the image/video transforms
            imgs = self.transform(imgs)
        return imgs

    def __len__(self):
        return self.data.shape[1]
