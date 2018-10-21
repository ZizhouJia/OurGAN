# -*- coding: UTF-8 -*-
import torch
import module
import dataset.mnist_color.mnist_color as mnist_color
import utils
import os

def reconstruction_loss(x1,x2):
    loss=torch.abs(x1-x2).mean()
    return loss

def D_real_loss(output):
    loss=torch.abs(output-1.0).mean()
    return loss

def D_fake_loss(output):
    loss=torch.abs(output).mean()
    return loss

def G_fake_loss(output):
    loss=torch.abs(output-1.0).mean()
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
    dis_x2_fake=dis_iamge(x2_fake)
    dis_feature_real=dis_feature(feature_real)
    dis_feature_fake_x1=dis_feature(diff1)
    dis_feature_fake_x2=dis_feature(diff2)

    return x1,x2,x1_fake,x2_fake,dis_x1_real,dis_x2_real,dis_x1_fake,dis_x2_fake,feature_real,dis_feature_fake_x1,dis_feature_fake_x2

def forward_and_get_loss(models,x1,x2,feature_real):
    x1,x2,x1_fake,x2_fake,dis_x1_real,dis_x2_real,
    dis_x1_fake,dis_x2_fake,feature_real,dis_feature_fake_x1,dis_feature_fake_x2=forward(models,x1,x2,feature_real)
    reconst_loss=reconstruction_loss(x1,x1_fake)+reconstruction_loss(x2,x2_fake)
    feature_D_loss=D_real_loss(feature_real)+D_fake_loss(dis_feature_fake_x1)+D_fake_loss(dis_feature_fake_x2)
    image_D_loss=D_fake_loss(x1_fake)+D_fake_loss(x2_fake)+D_real_loss(x1)+D_real_loss(x2)
    G_loss=G_fake_loss(x1_fake)+G_fake_loss(x2_fake)+G_fake_loss(dis_feature_fake_x1)+G_fake_loss(dis_feature_fake_x2)
    return reconst_loss,feature_D_loss,image_D_loss,G_loss

def backward(optimizers,reconst_loss,feature_D_loss,image_D_loss,G_loss):
    #这里暂且当做一个临时变量来使用
    param=0.1
    encoder_optimizer,decoder_optimizer,image_D_optimizer,feature_D_optimizer=optimizers
    total_G_loss=G_loss+param*reconst_loss
    total_G_loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    image_D_optimizer.zero_grad()
    feature_D_optimizer.zero_grad()
    feature_D_loss.backward()
    feature_D_optimizer.step()
    feature_D_optimizer.zero_grad()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    image_D_optimizer.backward()
    image_D_optimizer.zero_grad()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

def generate_optimizers(models,lr=0.001,weight_decay=0.0001,momentum=0.9):
    encoder,decoder,dis_image,dis_feature=models
    optimizers=[]
    for model in models:
        optimizer=torch.optim.SGD(model.parameters(),lr=lr,weight_decay=weight_decay,momentum=momentum)
        optimizers.append(optimizer)

def generate_models(is_cuda):
    models=[]
    encoder=module.GAN_module_mnist.encoder()
    models.append(encoder)
    decoder=module.GAN_module_mnist.decoder()
    models.append(decoder)
    image_dis=module.GAN_module_mnist.discriminator_for_image()
    models.append(image_dis)
    feature_dis=module.GAN_module_mnist.discriminator_for_difference()
    models.append(feature_dis)
    return models

def generate_dataset(batch_size=16):
    #创建数据集，这个数据集每次调用返回两张图片，一张为某种颜色的手写字，另外一张为另一种颜色的手写字
    mnist_loader=Data.DataLoader(mnist_color.minst_color(),batch_size=batch_size,shffle=True,num_workers=0)
    #创建一个随机噪声当做学习的中间特征的表达形式
    noise_loader=utils.data_provider.data_provider(utils.random_noise_producer.random_noise(),batch_size=batch_size)
    return mnist_loader,noise_loader

def save_models(models,save_path="checkpoints"):
    file_name=["encoder.pkl","decoder.pkl","image_dis.pkl","feature_dis.pkl"]
    for i in range(0,len(models)):
        torch.save(models[i].state_dict(),os.path.join(save_path,file_name[i]))

def restore_models(models,save_path="checkpoints"):
    file_name=["encoder.pkl","decoder.pkl","image_dis.pkl","feature_dis.pkl"]
    for i in range(0,len(models)):
        models[i].load_state_dict(torch.load(os.path.join(save_path,file_name[i])))


def report_loss(reconst_loss,feature_D_loss,image_D_loss,G_loss,step):
    print("In step %d  rstl:%.4f fdl:%.4f idl:%.4f gl:%.4f"
    %(step,reconst_loss.cpu().item(),feature_D_loss.cpu().item(),image_D_loss.cpu().item(),G_loss.cpu().item())


def train():
    is_cuda=False
    if(torch.cuda.is_available()):
        is_cuda=True

    models=generate_models()
    if(is_cuda):
        for i in range(0,len(models)):
            models[i]=models[i].cuda()

    optimizers=generate_optimizers(models)
    mnist_loader,noise_loader=generate_dataset()

    #epoch的次数，这里先写成变量，以后会加入到config文件中
    epoch=100

    for i in range(0,epoch):
        for step,(x1,x2) in enumerate(mnist_loader):
            noise=noise_loader.next()
            if(is_cuda):
                x1=x1.cuda()
                x2=x2.cuda()
                noise=noise.cuda()
            reconst_loss,feature_D_loss,image_D_loss,G_loss=forward_and_get_loss(models,x1,x2,noise)
            if(step%100==0):
                report_loss(reconst_loss,feature_D_loss,image_D_loss,G_loss,step)
            backward(optimizers,reconst_loss,feature_D_loss,image_D_loss,G_loss)
        if((epoch+1)%10==0):
            save_models(models)

def test_on_mnist:
    is_cuda=False
    if(torch.cuda.is_available()):
        is_cuda=True
    models=generate_models()
    if(is_cuda):
        for i in range(0,len(models)):
            models[i]=models[i].cuda()
    restore_models(models)
