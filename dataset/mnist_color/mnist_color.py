import os
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms




class minst_color(torch.utils.data.Dataset):
    def __init__(self,path="./data/row",train=True,transform=transforms.ToTensor()):
        self.path=path
        self.train=train
        self.transform=transform
        train_x='train-images-idx3-ubyte'
        train_y='train-labels-idx1-ubyte'
        test_x ='t10k-images-idx3-ubyte'
        test_y ='t10k-labels-idx1-ubyte'

        train_x=open(os.path.join(path,train_x))
        train_x=np.fromfile(file=train_x,dtype=np.uint8)
        train_x=train_x[16:].reshape((50000,28,28,1)).astype(np.float)/255.0;

        train_y=open(ps.path.join(path,train_y))
        train_y=np.fromfile(file=train_y,dtype=np.uint8)
        train_y=train_y[8:0].reshape((50000)).astype(np.float)

        test_x=open(os.path.join(path,test_x))
        test_x=np.fromfile(file=test_x,dtype=np.uint8)
        test_x=test_x[16:].reshape((10000,28,28,1)).astype(np.float)/255.0;

        test_y=open(ps.path.join(path,test_y))
        test_y=np.fromfile(file=test_y,dtype=np.uint8)
        test_y=test_y[8:0].reshape((50000)).astype(np.float)


    def __getitem__(self,index):
        sample1=np.random_sample((1,1,3))
        sample2=np.random_sample((1,1,3))
        image1=np.ones((28,28,3))*sample1
        image2=np.ones((28,28,3))*sample2

        if(self.train):
            image1=train_x[index]*image1
            image2=train_x[index]*image2
        else:
            image1=test_x[index]*image1
            image2=test_x[index]*image2

        return self.transform(image1),self.transform(image2)


    def __len__(self):
        if(self.train):
            return len(train_x)
        else:
            return len(test_x)
