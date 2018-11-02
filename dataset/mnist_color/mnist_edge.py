import os
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import cv2


def sobel(image):
    gray_lap=cv2.Laplacian(image,cv2.CV_16S,ksize=3)
    gray_lap=gray_lap.astype(np.float32)
    gray_lap=np.abs(gray_lap)
    gray_lap=gray_lap/np.max(gray_lap)*255.0
    gray_lap=gray_lap.astype(np.uint8)
    return gray_lap



class mnist_edge(torch.utils.data.Dataset):
    def __init__(self,path="./dataset/mnist_color/data/processed",train=True,transform=transforms.ToTensor()):
        self.path=path
        self.train=train
        self.transform=transform
        train_x='train-images-idx3-ubyte'
        train_y='train-labels-idx1-ubyte'
        test_x ='t10k-images-idx3-ubyte'
        test_y ='t10k-labels-idx1-ubyte'
        seed=123
        np.random.seed(seed)

        train_x=open(os.path.join(path,train_x))
        train_x=np.fromfile(file=train_x,dtype=np.uint8)
        self.train_x=train_x[16:].reshape((60000,28,28));
        for i in range(0,len(self.train_x)):
            self.train_x[i,:,:]=sobel(self.train_x[i,:,:])
        np.random.shuffle(self.train_x)
        self.train_x=self.train_x.reshape((60000,28,28,1)).astype(np.float32)/255.0

        test_x=open(os.path.join(path,test_x))
        test_x=np.fromfile(file=test_x,dtype=np.uint8)
        self.test_x=test_x[16:].reshape((10000,28,28));
        for i in range(0,len(self.test_x)):
            self.test_x[i,:,:]=sobel(self.test_x[i,:,:])
        np.random.shuffle(self.test_x)
        self.test_x=self.test_x.reshape((10000,28,28,1)).astype(np.float32)/255.0

    def __getitem__(self,index):
        image=None
        if(self.train):
            image=self.train_x[index]
        else:
            image=self.test_x[index]
        image=image*2-1

        return self.transform(image)


    def __len__(self):
        if(self.train):
            return len(self.train_x)
        else:
            return len(self.test_x)
