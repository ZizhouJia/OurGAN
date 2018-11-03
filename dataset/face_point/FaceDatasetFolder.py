import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import os

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
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images





class FaceDatasetFolder(torch.utils.data.Dataset):
    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        classes, class_to_idx = find_classes(root)
        samples = make_dataset(root, class_to_idx, extensions)
        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

#self.train = train
        self.transform = transform
        self.target_transform = target_transform




    def __getitem__(self,index):






    def __len__(self):
        if(self.train):
            return len(self.train)
        else:
            return len(self.test)
