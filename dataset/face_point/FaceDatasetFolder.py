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
    class_to_idx = {classes[i]:i for i in xrange(len(classes))}
    return classes, class_to_idx

def pil_loader(path):
    with open(path, 'rb') as f:
        img=Image.open(f)
        #print img.size
        return img.convert('RGB')

#get image info
def make_dataset(dir,class_to_idx,extentions):
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
                    img = pil_loader(path)
                    
                    #print(path)
                    #print img
                    item = (img, class_to_idx[target])
                    images.append(item)
    return images

#id->"0" all images
def get_class_items(samples, classes):
    classlen = len(classes)
    classitem = [[]]*classlen
    #print(samples)
    for i, (path, id) in enumerate(samples):
        item = (path, i)
        classitem[id].append(item)
    return classitem


class FaceDatasetFolder(torch.utils.data.Dataset):
    def __init__(self, root, loader=pil_loader, extensions=IMG_EXTENTIONS, transform=transform_img, target_transform=None):
        classes, class_to_idx = find_classes(root)
        samples = make_dataset(root, class_to_idx, extensions)
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


    def __getitem__(self, index):
        img, target = self.samples[index]
        coindex = random.randint(0, 37)
        (img2, index2) = self.classitem[target][coindex]
        while index2 == index:
            coindex = random.randint(0, 37)
            (img2, index2) = self.classitem[target][coindex]

        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img*2-1, img2*2-1) #, target





    def __len__(self):
        return len(self.samples)
