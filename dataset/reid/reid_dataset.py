import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import torchvision.transforms as transforms
import os
import random
import key_point_tools.key_point as key_point
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

from reid.utils.data import transforms

try:
    import cPickle as pickle
except ImportError:
    import pickle


normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_img = transforms.Compose([
                         transforms.RectScale(256, 128),
                         transforms.RandomSizedEarser(),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         normalizer,
                    ])


def pil_loader(path):
    with open(path, 'rb') as f:
        img=Image.open(f)
        #print img.size
        return img.convert('RGB')

#get image info
def make_dataset(dir):
    #idx->class
    idx_to_class=[]
    samples=[]

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            label=fname[0:4]
            if label not in idx_to_class:
                idx_to_class.append(label)

            index=idx_to_class.index(label)

            path = os.path.join(root, fname)
            img = pil_loader(path)
            item = (img,index)
            #print(item)
            samples.append(item)
            #sys.exit(0)
    return idx_to_class,samples

#id->"0" all images
def get_class_items(samples, classes):
    classlen = len(classes)
    classitem = [[] for i in range(classlen)]
    classidx_onehot=[]
    for i in range(classlen):
        onehot=np.zeros((classlen), dtype=np.int)
        onehot[i]=1
        classidx_onehot.append(onehot)
    for i, (img, idx) in enumerate(samples):
        item = (img ,i)
        classitem[idx].append(item)
    return classitem,classidx_onehot


class reid_dataset(torch.utils.data.Dataset):
    def __init__(self, root, loader=pil_loader,  transform=transform_img,load_data=False,mode="train"):

        if load_data == True:
            if mode=="train":
               loadfile=open('processdata/traindata.pkl','rb')
            elif mode=="test":
               loadfile=open('processdata/testdata.pkl','rb')
            elif mode=="query":
               loadfile=open('processdata/querydata.pkl','rb')
            samples=pickle.load(loadfile)
            classitem=pickle.load(loadfile)
            idx_to_class=pickle.load(loadfile)
            classidx_onehot=pickle.load(loadfile)
        else:
            idx_to_class,samples = make_dataset(root)
            classitem, classidx_onehot= get_class_items(samples, idx_to_class)
        self.root = root
        self.loader = loader

        self.classidx_onehot = classidx_onehot
        self.idx_to_class = idx_to_class
        self.samples = samples
        self.classitem = classitem
        self.mode=mode
        self.transform = transform

        if load_data == False:
            if mode=="train":
               loadfile=open('processdata/triandata.pkl','wb')
            elif mode=="test":
               loadfile=open('processdata/testdata.pkl','wb')
            elif mode=="query":
                loadfile=open('processdata/querydata.pkl','wb')
            pickle.dump(samples,loadfile,True)
            pickle.dump(classitem,loadfile,True)
            pickle.dump(idx_to_class,loadfile,True)
            pickle.dump(classidx_onehot,loadfile,True)



    def __getitem__(self, index):
        img,classidx = self.samples[index]

        if self.mode!="train":
            if self.transform is not None:
                img = self.transform(img)
            return (img*2-1,classidx)


        coindex = random.randint(0, len(self.classitem[classidx])-1)
        (img2, index2) = self.classitem[classidx][coindex]
        while index2 == index:
            coindex = random.randint(0, len(self.classitem[classidx])-1)
            (img2, index2) = self.classitem[classidx][coindex]

        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)
        classonehot=self.classidx_onehot[classidx]
        return (img*2-1, img2*2-1,classonehot)



    def __len__(self):
        return len(self.samples)
