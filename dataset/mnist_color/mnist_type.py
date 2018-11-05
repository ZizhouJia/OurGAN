import os
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import random


class minst_type(torch.utils.data.Dataset):
    def __init__(self,path="./dataset/mnist_color/data/processed",train=True,transform=transforms.ToTensor()):
        self.path=path
        self.train=train
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

        self.train=[]
        for i in range(0,10):
            self.train.append([])
        for j in range(0,60000):
            self.train[int(self.train_y[j])].append(self.train_x[j])


        test_x=open(os.path.join(path,test_x))
        test_x=np.fromfile(file=test_x,dtype=np.uint8)
        self.test_x=test_x[16:].reshape((10000,28,28,1)).astype(np.float32)/255.0

        test_y=open(os.path.join(path,test_y))
        test_y=np.fromfile(file=test_y,dtype=np.uint8)
        self.test_y=test_y[8:].reshape((10000))

        self.test=[]
        for i in range(0,10):
            self.test.append([])
        for j in range(0,10000):
            self.test[int(self.test_y[j])].append(self.test_x[j])

    def __getitem__(self,index):
        sample1=np.random.random((1,1,3)).astype(np.float32)
        sample2=np.random.random((1,1,3)).astype(np.float32)
        image1=np.ones((28,28,3),dtype=np.float32)*sample1
        image2=np.ones((28,28,3),dtype=np.float32)*sample2
        if(self.train):
            cls=index%10
            r1=random.randint(0,len(self.train[cls])-1)
            r2=random.randint(0,len(self.train[cls])-1)
            image1=self.train[cls][r1]*image1
            image2=self.train[cls][r2]*image2
        else:
            cls=index%10
            r1=random.randint(0,6000-1)
            r2=random.randint(0,6000-1)
            image1=self.test[cls][r1]*image1
            image2=self.test[cls][r2]*image2
        image1=image1*2-1
        image2=image2*2-1
        return self.transform(image1),self.transform(image2)


    def __len__(self):
        if(self.train):
            return len(self.train_x)
        else:
            return len(self.test_x)
