import os
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import random


class minst(torch.utils.data.Dataset):
    def __init__(self,path="./dataset/mnist_color/data/processed",train=True,transform=transforms.ToTensor()):
        self.path=path
        self.if_train=train
        self.transform=transform
        train_x='train-images-idx3-ubyte'
        train_y='train-labels-idx1-ubyte'
        test_x ='t10k-images-idx3-ubyte'
        test_y ='t10k-labels-idx1-ubyte'

        train_x=open(os.path.join(path,train_x))
        train_x=np.fromfile(file=train_x,dtype=np.uint8)
        self.train_x=train_x[16:].reshape((60000,28,28,1)).astype(np.float32)/255.0

        train_y=open(os.path.join(path,train_y))
        train_y=np.fromfile(file=train_y,dtype=np.uint8)
        self.train_y=train_y[8:].reshape((60000))

        test_x=open(os.path.join(path,test_x))
        test_x=np.fromfile(file=test_x,dtype=np.uint8)
        self.test_x=test_x[16:].reshape((10000,28,28,1)).astype(np.float32)/255.0

        test_y=open(os.path.join(path,test_y))
        test_y=np.fromfile(file=test_y,dtype=np.uint8)
        self.test_y=test_y[8:].reshape((10000))

    def __getitem__(self,index):
        image=None
        label=None
        if(self.if_train):
            image=self.train_x[index]
            label=self.train_y[index]
            temp=np.zeros((1))
            temp[0]=label
            label=torch.Tensor(temp).long()
        else:
            image=self.test_x[index]
            label=self.test_y[index]
            temp=np.zeros((1))
            temp[0]=label
            label=torch.Tensor(temp).long()
        return self.transform(image),label


    def __len__(self):
        if(self.if_train):
            return 10000
        else:
            return 1000
