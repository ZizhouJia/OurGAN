# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import model.GAN_module_mnist as GAN_module_mnist
import model.GAN_module_mnist_style as GAN_module_mnist_style
import dataset.mnist_color.mnist_color as mnist_color
import dataset.mnist_color.mnist_style as mnist_style
import dataset.mnist_color.mnist_edge as mnist_edge
import torch.utils.data as Data
import utils.data_provider as data_provider
import utils.random_noise_producer as random_noise_producer
<<<<<<< HEAD
import dataset.face_point.FaceDatasetFolder as FaceDatasetFolder
import math
=======
import dataset.face_point.FaceImageFolder as FaceImageFolder
>>>>>>> c641d38c47447134c9e89c47ec200f1b2bfef453


#weight initialization
def weights_init(init_type='default'):
    def init_func(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass

    return init_func


# loss function
def reconstruction_loss(x1,x2):
    loss=torch.abs(x1-x2).mean()
    return loss

def D_real_loss(output,loss_func="lsgan"):
    if(loss_func=="lsgan"):
        distance=(output-1.0)*(output-1.0)
        loss=distance.mean()
        return loss

    if(loss_func=="wgan"):
        squ=(output-1)
        squ=squ*squ
        return (-output).mean()+squ.mean()

    if(loss_func =="hinge"):
        real_loss =torch.functional.F.relu(1.0 - output).mean()
        return real_loss

def D_fake_loss(output,loss_func="lsgan"):
    if(loss_func=="lsgan"):
        distance=output*output
        loss=distance.mean()
        return loss

    if(loss_func=="wgan"):
        squ=(output)
        squ=squ*squ
        return output.mean()+squ.mean()

    if(loss_func =="hinge"):
        real_loss =torch.functional.F.relu(1.0 + output).mean()
        return real_loss


def G_fake_loss(output,loss_func="lsgan"):
    if(loss_func=="lsgan"):
        distance=(output-1)*(output-1)
        loss=distance.mean()
        return loss

    if(loss_func=="wgan"):
        squ=(output-1)
        squ=squ*squ
        return (-output).mean()+squ.mean()

    if(loss_func=="hinge"):
        return (-output).mean()


#optimizer
def generate_optimizers(models,lrs,optimizer_type="sgd",weight_decay=0.001):
    optimizers=[]
    if(optimizer_type=="sgd"):
        for i in range(0,len(models)):
            optimizer=torch.optim.SGD(models[i].parameters(),lr=lrs[i],weight_decay=weight_decay,momentum=0.9)
            # optimizer=nn.DataParallel(optimizer)
            optimizers.append(optimizer)

    if(optimizer_type=="adam"):
        for i in range(0,len(models)):
            optimizer=torch.optim.Adam(models[i].parameters(),lr=lrs[i],weight_decay=weight_decay,betas=(0.5, 0.999))
            # optimizer=nn.DataParallel(optimizer)
            optimizers.append(optimizer)
    return optimizers

#dataset
def generate_dataset(dataset_name,batch_size=32,train=True):
    #创建数据集，这个数据集每次调用返回两张图片，一张为某种颜色的手写字，另外一张为另一种颜色的手写字
    if(dataset_name=='mnist'):
        if(train):
            mnist_loader=Data.DataLoader(mnist_color.minst_color(path="dataset/mnist_color/data/raw/",train=True),batch_size=batch_size,shuffle=True,num_workers=0)
            #创建一个随机噪声当做学习的中间特征的表达形式
            noise_loader=data_provider.data_provider(random_noise_producer.random_noise(),batch_size=batch_size)
            return mnist_loader,noise_loader
        else:
            mnist_loader=Data.DataLoader(mnist_color.minst_color(path="./dataset/mnist_color/data/raw/",train=False),batch_size=batch_size,num_workers=0)
            noise_loader=data_provider.data_provider(random_noise_producer.random_noise(),batch_size=batch_size)
            return mnist_loader,noise_loader

    if(dataset_name=='mnist_style'):
        if(train):
            mnist_loader=Data.DataLoader(mnist_style.minst_style(path="dataset/mnist_color/data/raw/",train=True),batch_size=batch_size,shuffle=True,num_workers=0)
            mnist_edge_loader=data_provider.data_provider(mnist_edge.mnist_edge(path="dataset/mnist_color/data/raw/",train=False),batch_size=batch_size)
            return mnist_loader,mnist_edge_loader
        else:
            mnist_loader=Data.DataLoader(mnist_style.minst_style(path="dataset/mnist_color/data/raw/",train=False),batch_size=batch_size,shuffle=False,num_workers=0)
            mnist_edge_loader=data_provider.data_provider(mnist_edge.mnist_edge(path="dataset/mnist_color/data/raw/",train=False),batch_size=batch_size)
            return mnist_loader,mnist_edge_loader
        if(dataset_name=='face_point'):
            if(train):
                imagedatasets = FaceImageFolder.FaceImageFolder(root="dataset/face_point/data/train/")
                imageloader = Data.DataLoader(imagedatasets, batch_size=batch_size, shuffle=True, num_workers=0)
                mnist_edge_loader = data_provider.data_provider(mnist_edge.mnist_edge(path="dataset/mnist_color/data/raw/",train=False),batch_size=batch_size)
                return imageloader, mnist_edge_loader
            else:
                imagedatasets = FaceImageFolder.FaceImageFolder(root="dataset/face_point/data/test/")
                imageloader = Data.DataLoader(imagedatasets, batch_size=batch_size, shuffle=False, num_workers=0)
                mnist_edge_loader = data_provider.data_provider(mnist_edge.mnist_edge(path="dataset/mnist_color/data/raw/",train=False),batch_size=batch_size)
                return imageloader, mnist_edge_loader

def parallel(models):
    for i in range(0,len(models)):
        models[i]=nn.DataParallel(models[i])
    # parallel(models)
    return models

    if(dataset_name=='face_point'):
        if(train):
            imagedatasets = FaceDatasetFolder.FaceDatasetFolder(root="dataset/face_point/data/train/")
            imageloader = Data.DataLoader(imagedatasets, batch_size=batch_size, shuffle=True, num_workers=0)
            mnist_edge_loader = data_provider.data_provider(mnist_edge.mnist_edge(path="dataset/mnist_color/data/raw/",train=False),batch_size=batch_size)
            return imageloader, mnist_edge_loader
        else:
            imagedatasets = FaceDatasetFolder.FaceDatasetFolder(root="dataset/face_point/data/test/")
            imageloader = Data.DataLoader(imagedatasets, batch_size=batch_size, shuffle=False, num_workers=0)
            mnist_edge_loader = data_provider.data_provider(mnist_edge.mnist_edge(path="dataset/mnist_color/data/raw/",train=False),batch_size=batch_size)
            return imageloader, mnist_edge_loader


#models
def generate_models(model_name):
    if(model_name=='GAN_mnist'):
        models=[]
        encoder=GAN_module_mnist.encoder()
        models.append(encoder)
        decoder=GAN_module_mnist.decoder()
        models.append(decoder)
        image_dis=GAN_module_mnist.discriminator_for_image()
        models.append(image_dis)
        feature_dis=GAN_module_mnist.discriminator_for_difference()
        models.append(feature_dis)

    if(model_name=='GAN_mnist_style'):
        models=[]
        encoder=GAN_module_mnist_style.encoder()
        models.append(encoder)
        decoder=GAN_module_mnist_style.decoder()
        models.append(decoder)
        image_dis=GAN_module_mnist_style.discriminator_for_image()
        models.append(image_dis)
        feature_dis=GAN_module_mnist_style.discriminator_for_difference()
        models.append(feature_dis)
        return models

    if(model_name=='GAN_face_point'):
        models=[]
        encoder=GAN_module_mnist_style.encoder()
        models.append(encoder)
        decoder=GAN_module_mnist_style.decoder()
        models.append(decoder)
        image_dis=GAN_module_mnist_style.discriminator_for_image()
        models.append(image_dis)
        feature_dis=GAN_module_mnist_style.discriminator_for_difference()
        models.append(feature_dis)
        return models
