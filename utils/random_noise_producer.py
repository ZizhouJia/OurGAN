# -*- coding: UTF-8 -*-
import os
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

class random_noise(torch.utils.data.Dataset):
    def __init__(self,transform=transforms.ToTensor()):
        self.train_x=np.random.randn(100000,1,1,10).astype(np.float32)
        self.transform=transform


    def __getitem__(self,index):

        return self.transform(self.train_x[index])


    def __len__(self):
        return len(self.train_x)

def get_noise_numpy(batch_size=1):
    return np.random.randn(batch_size,10,1,1)

# def get_certain_noise():
