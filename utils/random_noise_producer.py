import os
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms




class random_noise(torch.utils.data.Dataset):
    def __init__(self,transform=transforms.ToTensor()):
        self.train_x=np.random.normal(100000,10,1,1)

    def __getitem__(self,index):

        return self.transform(self.train_x[index])


    def __len__(self):
        return self.train_x
