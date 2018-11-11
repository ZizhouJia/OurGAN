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
import matplotlib.pyplot as plt
try:
    import cPickle as pickle
except ImportError:
    import pickle

IMG_EXTENTIONS = [".jpg", ".jpeg", ".bmp", ".png", ".tif"]

transform_img = transforms.Compose([
                    transforms.CenterCrop(480),
                    transforms.Resize(64),
                    transforms.ToTensor(),
                    ])

#file is image?

def has_file_allowed_extension(filename,extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

#find image class
def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir,d))]
    classes.sort()
    class_to_idx = {classes[i]:i for i in range(len(classes))}
    return classes, class_to_idx

def pil_loader(path):
    with open(path, 'rb') as f:
        img=Image.open(f)
        #print img.size
        return img.convert('RGB')

#get image info
def make_dataset(dir,class_to_idx,extentions,marker):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extentions):
                    path = os.path.join(root, fname)
                    #print(path)
                    #print(len(path))
                    #print(path[len(path)-6:])
                    #img = pil_loader(path)
                    #print(path)
                    img = cv2.imread(path)
                    #print(img)
                    landmark_matrix=marker.get_key_point(img)
                    if landmark_matrix == None:
                        continue
                    #print(landmark_matrix)
                    #print(path)
                    feature_map=marker.write_feature_map(landmark_matrix,img.shape)
                    #print(feature_map.astype)
                    feature=Image.fromarray(feature_map[:,:,0])
                    #feature.save("out.jpg")
                    #print(feature)

                    #print img
                    imgp = pil_loader(path)
                    item = (imgp,feature, class_to_idx[target])

                    images.append(item)
                    #sys.exit(0)
    return images

#id->"0" all images
def get_class_items(samples, classes):
    classlen = len(classes)
    classitem = [[] for i in range(classlen)]
    #print(samples)
    for i, (img, feature,id) in enumerate(samples):
        item = (img,feature ,i)
        classitem[id].append(item)
    return classitem


class face_point_dataset(torch.utils.data.Dataset):
    def __init__(self, root, loader=pil_loader, extensions=IMG_EXTENTIONS, transform=transform_img, target_transform=None,load_data=False,train=True):
        self.marker=key_point.face_key_point_marker("key_point_tools/shape_predictor_68_face_landmarks.dat")
        classes, class_to_idx = find_classes(root)
        if load_data == True:
            if train==True:
               loadfile=open('processdata/triandata.pkl','rb')
            else:
               loadfile=open('processdata/testdata.pkl','rb')
            samples=pickle.load(loadfile)
            classitem=pickle.load(loadfile)
        else:
            samples = make_dataset(root, class_to_idx, extensions,self.marker)
            classitem = get_class_items(samples, classes)
        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.classitem = classitem

#self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if load_data == False:
            if train==True:
               loadfile=open('processdata/triandata.pkl','wb')
            else:
               loadfile=open('processdata/testdata.pkl','wb')
            pickle.dump(samples,loadfile,True)
            pickle.dump(classitem,loadfile,True)



    def __getitem__(self, index):
        img,feature, target = self.samples[index]

        coindex = random.randint(0, len(self.classitem[target])-1)
        (img2,feature2, index2) = self.classitem[target][coindex]
        while index2 == index:
            coindex = random.randint(0, len(self.classitem[target])-1)
            (img2,feature2, index2) = self.classitem[target][coindex]

        coindex2 = random.randint(0, len(self.classitem[target])-1)
        (img3,feature3, index3) = self.classitem[target][coindex2]
        while index3 == index or index3 ==index2:
            coindex2 = random.randint(0, len(self.classitem[target])-1)
            (img3,feature3, index3) = self.classitem[target][coindex2]

        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)
            feature3 = self.transform(feature3)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img*2-1, img2*2-1,feature3*2-1) #, target





    def __len__(self):
        return len(self.samples)
