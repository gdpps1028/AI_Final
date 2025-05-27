import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import numpy as np
from torchvision import transforms
import random

def erase_random_patch(img, size_range=(10, 25)):
    img = np.array(img)
    h, w = img.shape
    x1 = np.random.randint(0, w - size_range[1])
    y1 = np.random.randint(0, h - size_range[1])
    x2 = x1 + np.random.randint(*size_range)
    y2 = y1 + np.random.randint(*size_range)
    img[y1:y2, x1:x2] = 0
    return Image.fromarray(img)

class StrokeCharDataset(Dataset):
    def __init__(self, samples, transform=None, train=True):
        self.samples = samples
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        if self.train and np.random.rand() < 0.3:
            img = erase_random_patch(img)
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, label