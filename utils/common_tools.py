# -*- coding: UTF-8 -*-
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import model.GAN_module_mnist as GAN_module_mnist
import model.GAN_module_mnist_style as GAN_module_mnist_style
import model.GAN_module_face_point as GAN_module_face_point
import model.GAN_module_reid as GAN_module_reid
import dataset.mnist_color.mnist_color as mnist_color
import dataset.mnist_color.mnist_style as mnist_style
import dataset.mnist_color.mnist_type as mnist_type
import dataset.mnist_color.mnist_edge as mnist_edge
import dataset.mnist_color.mnist as mnist
import torch.utils.data as Data
import utils.data_provider as data_provider
import utils.random_noise_producer as random_noise_producer

# import dataset.face_point.FaceDatasetFolder as FaceDatasetFolder
import dataset.reid.reid_dataset as reid_dataset
# import dataset.face_point.face_point_dataset as face_point_dataset

import math



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


def verify_loss(output,label):
    e=1e-8
    # loss=-(label)*torch.log(output+e)-(1-label)*torch.log(1-output+e)
    loss=F.cross_entropy(output,label)
    # loss=loss.mean()
    # loss=label*output-(1-label)*output/(torch.sum(1-label)+1)
    return loss#.mean()


def feature_same_loss(x1,x2):
    distance=F.normalize(x1)-F.normalize(x2)
    return torch.sum((distance*distance).view(distance.size()[0],-1),1).mean()

def D_classify_loss(output,target):
    e=1e-8
    # loss=target*torch.log(output)+(1-target)*torch.log(1-output)
    output=torch.exp(output)
    total=torch.sum(output,1).view(output.size()[0],1)
    output=torch.log(output/total+e)
    #print(target)
    #print(output)
    output=torch.sum(-target*output,1)
    return output.mean()

def G_classify_loss(output):
    e=1e-8
    output=torch.exp(output)
    total=torch.sum(output,1).view(output.size()[0],1)
    output=torch.log(output/total+e)
    output=torch.mean(-output,1)
    return output.mean()

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
        return (-output).mean()

    if(loss_func =="hinge"):
        real_loss =torch.functional.F.relu(1.0 - output).mean()
        return real_loss

def D_fake_loss(output,loss_func="lsgan"):
    if(loss_func=="lsgan"):
        distance=output*output
        loss=distance.mean()
        return loss

    if(loss_func=="wgan"):
        return output.mean()

    if(loss_func =="hinge"):
        real_loss =torch.functional.F.relu(1.0 + output).mean()
        return real_loss


def G_fake_loss(output,loss_func="lsgan"):
    if(loss_func=="lsgan"):
        distance=(output-1)*(output-1)
        loss=distance.mean()
        return loss

    if(loss_func=="wgan"):
        return (-output).mean()

    if(loss_func=="hinge"):
        return (-output).mean()


#optimizer
def generate_optimizers(models,lrs,optimizer_type="sgd",weight_decay=0.001):
    optimizers=[]
    if(optimizer_type=="sgd"):
        for i in range(0,len(models)):
            if(i==2 or i==3):
                optimizer=torch.optim.Adam(models[i].parameters(),lr=lrs[i],weight_decay=weight_decay,betas=(0.5, 0.999))
                optimizers.append(optimizer)
            else:
                optimizer=torch.optim.SGD(models[i].parameters(),lr=lrs[i],weight_decay=weight_decay,momentum=0.9)
                optimizers.append(optimizer)

    if(optimizer_type=="adam"):
        for i in range(0,len(models)):
            optimizer=torch.optim.Adam(models[i].parameters(),lr=lrs[i],weight_decay=weight_decay,betas=(0.5, 0.999))
            # optimizer=nn.DataParallel(optimizer)
            optimizers.append(optimizer)

    return optimizers

#dataset
def generate_dataset(dataset_name,batch_size=32,train=True,test_cross_class=False):
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

    # if(dataset_name=='face_point'):
    #     print("loading dataset...")
    #     if(train):
    #         feature_datasets = face_point_dataset.face_point_dataset(root="dataset/face_point/data/train/",load_data=True,train=True)
    #         feature_loader = Data.DataLoader(feature_datasets, batch_size=batch_size, shuffle=False, num_workers=0)
    #
    #         return  feature_loader
    #     else:
    #         feature_datasets = face_point_dataset.face_point_dataset(root="dataset/face_point/data/test/",load_data=True,train=False)
    #         feature_loader = Data.DataLoader(feature_datasets, batch_size=batch_size, shuffle=False, num_workers=0)
    #         return feature_loader


    if(dataset_name=='mnist_type'):
        if(train):
            mnist_loader=Data.DataLoader(mnist_type.minst_type(path="dataset/mnist_color/data/raw/",train=True),batch_size=batch_size,shuffle=True,num_workers=0)
            #创建一个随机噪声当做学习的中间特征的表达形式
            noise_loader=data_provider.data_provider(random_noise_producer.random_noise(),batch_size=batch_size)
            return mnist_loader
        else:
            query_loader=Data.DataLoader(mnist.minst(path="./dataset/mnist_color/data/raw/",train=False),batch_size=batch_size,num_workers=0)
            test_loader=Data.DataLoader(mnist.minst(path="./dataset/mnist_color/data/raw/",train=True),batch_size=batch_size,num_workers=0)
            return query_loader,test_loader

    if(dataset_name=='DukeMTMC-reID'):
        print("loading dataset...")
        if(train):
            feature_datasets = reid_dataset.reid_dataset(root="dataset/reid/DukeMTMC-reID/bounding_box_train/",load_data=True,mode="train")
            feature_loader = Data.DataLoader(feature_datasets, batch_size=batch_size, shuffle=True, num_workers=0)
            return  feature_loader
        else:
            test_datasets = reid_dataset.reid_dataset(root="dataset/reid/DukeMTMC-reID/bounding_box_test/",load_data=True,mode="test")
            test_loader = Data.DataLoader(test_datasets, batch_size=batch_size, shuffle=False, num_workers=0)
            query_datasets = reid_dataset.reid_dataset(root="dataset/reid/DukeMTMC-reID/query/",load_data=True,mode="query")
            query_loader = Data.DataLoader(query_datasets, batch_size=batch_size, shuffle=False, num_workers=0)
            return query_loader,test_loader




def parallel(models):
    for i in range(0,len(models)):
        models[i]=nn.DataParallel(models[i])
    # parallel(models)
    return models

#models
def generate_models(model_name):
    if(model_name=='GAN_mnist'):
        models=[]
        encoder=GAN_module_mnist.encoder()
        models.append(encoder.cuda())
        decoder=GAN_module_mnist.decoder()
        models.append(decoder.cuda())
        image_dis=GAN_module_mnist.discriminator_for_image()
        models.append(image_dis.cuda())
        feature_dis=GAN_module_mnist.discriminator_for_difference()
        models.append(feature_dis.cuda())
        verifier=GAN_module_mnist.verification_classifier(10,32)
        models.append(verifier.cuda())


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
        encoder=GAN_module_face_point.encoder()
        models.append(encoder)
        decoder=GAN_module_face_point.decoder()
        models.append(decoder)
        image_dis=GAN_module_face_point.discriminator_for_image()
        models.append(image_dis)
        feature_dis=GAN_module_face_point.discriminator_for_difference()
        models.append(feature_dis)

    if(model_name=='GAN_Duke'):
        models=[]
        encoder=GAN_module_reid.encoder()
        models.append(encoder.cuda())
        decoder=GAN_module_reid.decoder()
        models.append(decoder.cuda())
        image_dis=GAN_module_reid.discriminator_for_image()
        models.append(image_dis.cuda())
        feature_dis=GAN_module_reid.discriminator_for_difference()
        models.append(feature_dis.cuda())
        verifier_class=GAN_module_reid.verification_classifier(702,1024)
        models.append(verifier_class.cuda())
    return models
