import os
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms


class minst_style(torch.utils.data.Dataset):
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
        self.train_x=train_x[16:].reshape((60000,28,28,1)).astype(np.float32)/255.0;
        seed1=666 #random seed
        seed2=555 #random seed
        self.train_x_1=self.train_x.copy()
        self.train_x_2=self.train_x.copy()
        np.random.seed(seed1)
        np.random.shuffle(self.train_x_1)
        np.random.seed(seed2)
        np.random.shuffle(self.train_x_2)

        test_x=open(os.path.join(path,test_x))
        test_x=np.fromfile(file=test_x,dtype=np.uint8)
        self.test_x=test_x[16:].reshape((10000,28,28,1)).astype(np.float32)/255.0;
        self.test_x_1=self.test_x.copy()
        self.test_x_2=self.test_x.copy()
        np.random.seed(seed1)
        np.random.shuffle(self.test_x_1)
        np.random.seed(seed2)
        np.random.shuffle(self.test_x_2)

    def __getitem__(self,index):
        sample=np.random.random((1,1,3)).astype(np.float32)
        color=np.ones((28,28,3),dtype=np.float32)*sample
        image1=None
        image2=None
        if(self.train):
            image1=self.train_x_1[index]*color
            image2=self.train_x_2[index]*color
        else:
            image1=self.test_x_1[index]*color
            image2=self.test_x_2[index]*color
        image1=image1*2-1
        image2=image2*2-1

        return self.transform(image1),self.transform(image2)


    def __len__(self):
        if(self.train):
            return len(self.train_x)
        else:
            return len(self.test_x)
