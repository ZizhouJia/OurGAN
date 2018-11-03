import FaceDatasetFolder
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

IMG_EXTENTIONS = [".jpg", ".jpeg", ".bmp", ".png", ".tif"]

transform_img = transforms.Compose([
                    transforms.CenterCrop(480),
                    transforms.Resize(28),
                    transforms.ToTensor(),
                    ])


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class FaceImageFolder(FaceDatasetFolder.FaceDatasetFolder):
    def __init__(self, root, transform=transform_img, target_transform=None,
                 loader=pil_loader):
        super(FaceImageFolder, self).__init__(root, loader, IMG_EXTENTIONS, transform=transform, target_transform=target_transform)
        self.imgs = self.samples
        self.transforms = transform
        #self.target_transform = targe_transform
