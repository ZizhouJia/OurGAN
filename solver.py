# -*- coding: UTF-8 -*-
import torch
import model.GAN_module_mnist as GAN_module_mnist
import dataset.mnist_color.mnist_color as mnist_color
import torch.utils.data as Data
import utils.data_provider as data_provider
import utils.random_noise_producer as random_noise_producer
import os
import cv2
import numpy as np
import click

def reconstruction_loss(x1,x2):
    loss=torch.abs(x1-x2).mean()
    return loss

def D_real_loss(output):
    distance=(output-1.0)*(output-1.0)
    loss=distance.mean()
    return loss

def D_fake_loss(output):
    distance=output*output
    loss=distance.mean()
    return loss

def G_fake_loss(output):
    distance=(output-1)*(output-1)
    loss=distance.mean()
    return loss

def forward(models,x1,x2,feature_real):
    encoder,decoder,dis_image,dis_feature=models
    media1,diff1=encoder(x1)
    media2,diff2=encoder(x2)
    x1_fake=decoder(media2,diff1)
    x2_fake=decoder(media1,diff2)

    dis_x1_real=dis_image(x1)
    dis_x2_real=dis_image(x2)
    dis_x1_fake=dis_image(x1_fake)
    dis_x2_fake=dis_image(x2_fake)
    dis_feature_real=dis_feature(feature_real)
    dis_feature_fake_x1=dis_feature(diff1)
    dis_feature_fake_x2=dis_feature(diff2)

    return x1,x2,x1_fake,x2_fake,dis_x1_real,dis_x2_real,dis_x1_fake,dis_x2_fake,feature_real,dis_feature_real,dis_feature_fake_x1,dis_feature_fake_x2

def forward_and_get_loss(models,x1,x2,feature_real,step):
    x1,x2,x1_fake,x2_fake,dis_x1_real,dis_x2_real,dis_x1_fake,dis_x2_fake,feature_real,dis_feature_real,dis_feature_fake_x1,dis_feature_fake_x2=forward(models,x1,x2,feature_real)
    reconst_loss=reconstruction_loss(x1,x1_fake)+reconstruction_loss(x2,x2_fake)
    feature_D_loss=D_real_loss(dis_feature_real)+D_fake_loss(dis_feature_fake_x1)+D_fake_loss(dis_feature_fake_x2)
    image_D_loss=D_real_loss(dis_x1_real)+D_real_loss(dis_x2_real)+D_fake_loss(dis_x1_fake)+D_fake_loss(dis_x2_fake)

    G_image_loss=G_fake_loss(dis_x1_fake)+G_fake_loss(dis_x2_fake)
    G_feature_loss=G_fake_loss(dis_feature_fake_x1)+G_fake_loss(dis_feature_fake_x2)
    return reconst_loss,feature_D_loss,image_D_loss,G_image_loss,G_feature_loss

def zero_grad_for_all(optimizers):
    for optimizer in optimizers:
        optimizer.zero_grad()



def generate_optimizers(models,lr=[0.001,0.001,0.001,0.001],weight_decay=0.0001,momentum=0.9):
    encoder,decoder,dis_image,dis_feature=models
    optimizers=[]
    for i in range(0,len(models)):
        optimizer=torch.optim.SGD(models[i].parameters(),lr=lr[i],weight_decay=weight_decay,momentum=momentum)
        optimizers.append(optimizer)
    return optimizers


def generate_models():
    models=[]
    encoder=GAN_module_mnist.encoder()
    models.append(encoder)
    decoder=GAN_module_mnist.decoder()
    models.append(decoder)
    image_dis=GAN_module_mnist.discriminator_for_image()
    models.append(image_dis)
    feature_dis=GAN_module_mnist.discriminator_for_difference()
    models.append(feature_dis)
    return models

def generate_dataset(batch_size=2):
    #创建数据集，这个数据集每次调用返回两张图片，一张为某种颜色的手写字，另外一张为另一种颜色的手写字
    mnist_loader=Data.DataLoader(mnist_color.minst_color(path="dataset/mnist_color/data/raw/"),batch_size=batch_size,shuffle=True,num_workers=0)
    #创建一个随机噪声当做学习的中间特征的表达形式
    noise_loader=data_provider.data_provider(random_noise_producer.random_noise(),batch_size=batch_size)
    return mnist_loader,noise_loader

def save_models(models,save_path="checkpoints"):
    file_name=["encoder.pkl","decoder.pkl","image_dis.pkl","feature_dis.pkl"]
    for i in range(0,len(models)):
        torch.save(models[i].state_dict(),os.path.join(save_path,file_name[i]))

def restore_models(models,save_path="checkpoints"):
    file_name=["encoder.pkl","decoder.pkl","image_dis.pkl","feature_dis.pkl"]
    for i in range(0,len(models)):
        models[i].load_state_dict(torch.load(os.path.join(save_path,file_name[i])))


def report_loss(reconst_loss,feature_D_loss,image_D_loss,G_image_loss,G_feature_loss,step):
    print("In step %d  rstl:%.4f dfl:%.4f dil:%.4f gil:%.4f gfl:%.4f"%(step,reconst_loss.cpu().item(),feature_D_loss.cpu().item(),
    image_D_loss.cpu().item(),G_image_loss.cpu().item(),G_feature_loss.cpu().item()))



@click.command()
def train():
    is_cuda=False
    if(torch.cuda.is_available()):
        is_cuda=True
        print("cuda is available, current is cuda mode")
    else:
        print("cuda is unavailable, current is cpu mode")

    models=generate_models()
    if(is_cuda):
        for i in range(0,len(models)):
            models[i]=models[i].cuda()

    optimizers=generate_optimizers(models)
    mnist_loader,noise_loader=generate_dataset(batch_size=16)

    #epoch的次数，这里先写成变量，以后会加入到config文件中
    epoch=100
    encoder_optimizer,decoder_optimizer,image_D_optimizer,feature_D_optimizer=optimizers
    for i in range(0,epoch):
        print("begin the epoch : %d"%(epoch))
        for step,(x1,x2) in enumerate(mnist_loader):
            noise=noise_loader.next()
            if(is_cuda):
                x1=x1.cuda()
                x2=x2.cuda()
                noise=noise.cuda()
            #开始训练过程
            #先更新image discriminator
            reconst_loss,feature_D_loss,image_D_loss,G_image_loss,G_feature_loss=forward_and_get_loss(models,x1,x2,noise,step)
            image_D_loss.backward(retain_graph=True)
            image_D_optimizer.step()
            zero_grad_for_all(optimizers)

            #再更新feature discriminator
            feature_D_loss.backward(retain_graph=True)
            feature_D_optimizer.step()
            zero_grad_for_all(optimizers)

            #最后更新Generator
            total_loss=reconst_loss*1.0+G_image_loss+G_feature_loss
            total_loss.backward()
            decoder_optimizer.step()
            encoder_optimizer.step()
            zero_grad_for_all(optimizers)

            if(step%100==0):
                report_loss(reconst_loss,feature_D_loss,image_D_loss,G_image_loss,G_feature_loss,step)
        if((epoch+1)%10==0):
            save_models(models)

@click.command()
def test_on_mnist(file_output_path="test_output"):
    is_cuda=False
    if(torch.cuda.is_available()):
        is_cuda=True
        print("cuda is available, current is cuda mode")
    else:
        print("cuda is unavailable, current is cpu mode")
    models=generate_models()
    if(is_cuda):
        for i in range(0,len(models)):
            models[i]=models[i].cuda()
    restore_models(models)
    encoder=models[0]
    decoder=models[1]

    mnist_loader=Data.DataLoader(mnist_color.minst_color(train=False),batch_size=100,shffle=False,num_workers=0)

    for step,(x1,x2) in enumerate(mnist_color):
        noise=torch.Tensor(utils.random_noise_producer.get_noise_numpy(100))
        if(is_cuda):
            x1=x1.cuda()
            x2=x2.cuda()
            noise=noise.cuda()
        conv1,diff1=encoder(x1)
        conv2,diff2=encoder(x2)
        x_new1=decoder(conv1,noise)
        x_new2=decoder(conv2,noise)
        x1=x1.transpose(0,2,3,1).cpu().numpy()
        x2=x2.transpose(0,2,3,1).cpu().numpy()
        x_new1=x_new1.transpose(0,2,3,1).cpu().numpy()
        x_new2=x_new2.transpose(0,2,3,1).cpu().numpy()

        for j in range(0,100):
            image=np.zeros((x1.shape[2]*2,x1.shape[3]*2,3))
            image[0:x1.shape[2],0:x1.shape[3],:]=(x1[j,:,:,:]+1)/2
            image[0:x1.shape[2],x1.shape[3]:x1.shape[3]*2,:]=(x2[j,:,:,:]+1)/2
            image[x1.shape[2]:x1.shape[2]*2,0:x1.shape[3],:]=x_new1[j,:,:,:]
            image[x1.shape[2]:x1.shape[2]*2,x1.shape[3]:x1.shape[3]*2,:]=x_new2[j,:,:,:]
            cv2.imwrite(image,os.path.join(file_output_path,"test_"+str(step*100+j)))

@click.group()
def main():
    pass

if __name__ == '__main__':
    main.add_command(train)
    main.add_command(test_on_mnist)
    main()
