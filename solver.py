# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn

from utils.common_tools import *
import os
import cv2
import numpy as np
import click

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

def forward_and_get_loss(models,x1,x2,feature_real,step,train_method):
    x1,x2,x1_fake,x2_fake,dis_x1_real,dis_x2_real,dis_x1_fake,dis_x2_fake,feature_real,dis_feature_real,dis_feature_fake_x1,dis_feature_fake_x2=forward(models,x1,x2,feature_real)
    reconst_loss=reconstruction_loss(x1,x1_fake)+reconstruction_loss(x2,x2_fake)
    feature_D_loss=D_real_loss(dis_feature_real,train_method)+(D_fake_loss(dis_feature_fake_x1,train_method)+D_fake_loss(dis_feature_fake_x2,train_method))/2
    image_D_loss=D_real_loss(dis_x1_real,train_method)+D_real_loss(dis_x2_real,train_method)+D_fake_loss(dis_x1_fake,train_method)+D_fake_loss(dis_x2_fake,train_method)

    G_image_loss=G_fake_loss(dis_x1_fake,train_method)+G_fake_loss(dis_x2_fake,train_method)
    G_feature_loss=G_fake_loss(dis_feature_fake_x1,train_method)+G_fake_loss(dis_feature_fake_x2,train_method)
    return reconst_loss,feature_D_loss,image_D_loss,G_image_loss,G_feature_loss

def zero_grad_for_all(optimizers):
    for optimizer in optimizers:
        optimizer.zero_grad()

def init_models(models,init_type="default"):
    init_func=weights_init(init_type)
    for model in models:
        model.apply(init_func)

def save_models(models,model_name,dataset_name,save_path="checkpoints"):
    path=save_path
    if(not os.path.exists(path)):
        os.mkdir(path)
    path=os.path.join(path,model_name)

    if(not os.path.exists(path)):
        os.mkdir(path)
    path=os.path.join(path,dataset_name)

    if(not os.path.exists(path)):
        os.mkdir(path)

    file_name=["encoder.pkl","decoder.pkl","image_dis.pkl","feature_dis.pkl"]
    for i in range(0,len(models)):
        torch.save(models[i].state_dict(),os.path.join(path,file_name[i]))


def restore_models(models,model_name,dataset_name,save_path="checkpoints"):
    path=save_path
    if(not os.path.exists(path)):
        os.mkdir(path)
    path=os.path.join(path,model_name)

    if(not os.path.exists(path)):
        os.mkdir(path)
    path=os.path.join(path,dataset_name)

    if(not os.path.exists(path)):
        os.mkdir(path)

    file_name=["encoder.pkl","decoder.pkl","image_dis.pkl","feature_dis.pkl"]
    for i in range(0,len(models)):
        models[i].load_state_dict(torch.load(os.path.join(path,file_name[i])))


def report_loss(reconst_loss,feature_D_loss,image_D_loss,G_image_loss,G_feature_loss,step):
    print("In step %d  rstl:%.4f dfl:%.4f dil:%.4f gil:%.4f gfl:%.4f"%(step,reconst_loss.cpu().item(),feature_D_loss.cpu().item(),
    image_D_loss.cpu().item(),G_image_loss.cpu().item(),G_feature_loss.cpu().item()))


def train_eval_switch(models,train=True):
    if(train):
        for model in models:
            model.train()
    else:
        for model in models:
            model.eval()




@click.command()
@click.option('--batch_size',default=32,type=int, help="the batch size of train")
@click.option('--epoch',default=100,type=int, help="the total epoch of train")
@click.option('--dataset_name',default="mnist_style",type=click.Choice(["mnist","mnist_style","face_point"]),help="the string that defines the current dataset use")
@click.option('--model_name',default="GAN_mnist_style",type=click.Choice(["GAN_mnist","GAN_mnist_style","GAN_face_point"]),help="the string that  defines the current model use")
@click.option('--learning_rate',default=[0.0001,0.0001,0.0001,0.0001],nargs=4,type=float,help="the learning_rate of the four optimizer")
@click.option('--reconst_param',default=10.0,type=float,help="the reconstion loss coefficient")
@click.option('--image_d_loss_param',default=1.0,type=float,help="the image discriminator loss coefficient")
@click.option('--feature_d_loss_param',default=1.0,type=float,help="the feature discriminator loss coefficient")
@click.option('--model_save_path',default="checkpoints/",type=str,help="the model save path")
@click.option('--train_method',default="lsgan",type=click.Choice(["lsgan","wgan","hinge"]),help="the loss type of the train")
@click.option('--init_weight_type',default="default",type=click.Choice(["default","gaussian","xavier","kaiming","orthogonal"]),help="the loss type of the train")
@click.option('--optimizer_type',default="adam",type=click.Choice(["adam","sgd"]),help="the loss type of the train")
def train(batch_size,epoch,dataset_name,model_name,learning_rate,reconst_param,image_d_loss_param,feature_d_loss_param,model_save_path,train_method,init_weight_type,optimizer_type):
    is_cuda=False
    if(torch.cuda.is_available()):
        is_cuda=True
        print("cuda is available, current is cuda mode")
    else:
        print("cuda is unavailable, current is cpu mode")

    models=generate_models(model_name)
    init_models(models,init_weight_type)
    train_eval_switch(models)

    if(is_cuda):
        for i in range(0,len(models)):
            models[i]=models[i].cuda()

    optimizers=generate_optimizers(models,learning_rate,optimizer_type)
    mnist_loader,noise_loader=generate_dataset(dataset_name,batch_size,train=True)

    #epoch的次数，这里先写成变量，以后会加入到config文件中
    encoder_optimizer,decoder_optimizer,image_D_optimizer,feature_D_optimizer=optimizers
    for i in range(0,epoch):
        if(i%1==0):
            print("saving model....")
            save_models(models,model_name,dataset_name,model_save_path)
            print("save model succeed")

        print("begin the epoch : %d"%(i))
        for step,(x1,x2) in enumerate(mnist_loader):
            #print x1.size()
            noise=noise_loader.next()
            if(is_cuda):
                x1=x1.cuda()
                x2=x2.cuda()
                noise=noise.cuda()
            #开始训练过程
            #先更新image discriminator
            reconst_loss,feature_D_loss,image_D_loss,G_image_loss,G_feature_loss=forward_and_get_loss(models,x1,x2,noise,step,train_method)
            image_D_loss.backward(retain_graph=True)
            image_D_optimizer.step()
            zero_grad_for_all(optimizers)

            #再更新feature discriminator
            feature_D_loss.backward(retain_graph=True)
            feature_D_optimizer.step()
            zero_grad_for_all(optimizers)

            #最后更新Generator
            total_loss=reconst_loss*reconst_param+G_image_loss*image_d_loss_param+G_feature_loss*feature_d_loss_param
            total_loss.backward()
            decoder_optimizer.step()
            encoder_optimizer.step()
            zero_grad_for_all(optimizers)

            if(step%100==0):
                report_loss(reconst_loss,feature_D_loss,image_D_loss,G_image_loss,G_feature_loss,step)


@click.command()
@click.option('--batch_size',default=100,type=int, help="the batch size of the test")
@click.option('--dataset_name',default="mnist",type=click.Choice(["mnist","mnist_style","face_point"]),help="the string that defines the current dataset use")
@click.option('--model_name',default="GAN_mnist",type=click.Choice(["GAN_mnist","GAN_mnist_style","GAN_face_point"]),help="the string that  defines the current model use")
@click.option('--model_save_path',default="checkpoints/",type=str,help="the model save path")
@click.option('--file_save_path',default="test_output",type=str,help="the model save path")
def test_color(batch_size,dataset_name,model_name,model_save_path,file_save_path):
    is_cuda=False
    if(torch.cuda.is_available()):
        is_cuda=True
        print("cuda is available, current is cuda mode")
    else:
        print("cuda is unavailable, current is cpu mode")
    models=generate_models(model_name)
    train_eval_switch(models,False)

    if(is_cuda):
        for i in range(0,len(models)):
            models[i]=models[i].cuda()
    print("restoring models....")
    restore_models(models,model_name,dataset_name,model_save_path)
    print("restore models succeed")
    encoder=models[0]
    decoder=models[1]

    mnist_loader,noise_loader=generate_dataset(dataset_name,batch_size,train=False)


    for step,(x1,x2) in enumerate(mnist_loader):
        noise=noise_loader.next()
        if(is_cuda):
            x1=x1.cuda()
            x2=x2.cuda()
            noise=noise.cuda()

        same1,diff1=encoder(x1)
        same2,diff2=encoder(x2)
        x_new1=decoder(same1,noise)
        x_new2=decoder(same2,noise)
        x1=x1.permute(0,2,3,1).cpu().numpy()
        x2=x2.permute(0,2,3,1).cpu().numpy()
        x_new1=x_new1.permute(0,2,3,1).cpu().detach().numpy()
        x_new2=x_new2.permute(0,2,3,1).cpu().detach().numpy()

        for j in range(0,100):
            image=np.zeros((x1.shape[1]*2,x1.shape[2]*2,3))
            image[0:x1.shape[1],0:x1.shape[2],:]=(x1[j,:,:,:]+1)/2
            image[0:x1.shape[1],x1.shape[2]:x1.shape[2]*2,:]=(x2[j,:,:,:]+1)/2

            image[x1.shape[1]:x1.shape[1]*2,0:x1.shape[2],:]=(x_new1[j,:,:,:]+1)/2
            image[x1.shape[1]:x1.shape[1]*2,x1.shape[2]:x1.shape[2]*2,:]=(x_new2[j,:,:,:]+1)/2

            image=image*255
            image=image.astype(np.int32)
            cv2.imwrite(os.path.join(file_save_path,"test_"+str(step*100+j)+".jpg"),image)
        break


@click.command()
@click.option('--batch_size',default=100,type=int, help="the batch size of the test")
@click.option('--dataset_name',default="mnist_style",type=click.Choice(["mnist","mnist_style","face_point"]),help="the string that defines the current dataset use")
@click.option('--model_name',default="GAN_mnist_style",type=click.Choice(["GAN_mnist","GAN_mnist_style","GAN_face_point"]),help="the string that  defines the current model use")
@click.option('--model_save_path',default="checkpoints/",type=str,help="the model save path")
@click.option('--file_save_path',default="test_output",type=str,help="the model save path")
def test_edge(batch_size,dataset_name,model_name,model_save_path,file_save_path):
    is_cuda=False
    if(torch.cuda.is_available()):
        is_cuda=True
        print("cuda is available, current is cuda mode")
    else:
        print("cuda is unavailable, current is cpu mode")
    models=generate_models(model_name)
    train_eval_switch(models,False)

    if(is_cuda):
        for i in range(0,len(models)):
            models[i]=models[i].cuda()
    print("restoring models....")
    restore_models(models,model_name,dataset_name,model_save_path)
    print("restore models succeed")
    encoder=models[0]
    decoder=models[1]

    mnist_loader,edge_loader=generate_dataset(dataset_name,batch_size,train=False)


    for step,(x1,x2) in enumerate(mnist_loader):
        edge=edge_loader.next()
        if(is_cuda):
            x1=x1.cuda()
            x2=x2.cuda()
            edge=edge.cuda()

        same1,diff1=encoder(x1)
        same2,diff2=encoder(x2)
        x_new1=decoder(same1,diff1)
        x_new2=decoder(same2,edge)
        x1=x1.permute(0,2,3,1).cpu().numpy()
        x2=x2.permute(0,2,3,1).cpu().numpy()
        x_new1=x_new1.permute(0,2,3,1).cpu().detach().numpy()
        x_new2=x_new2.permute(0,2,3,1).cpu().detach().numpy()
        diff1=diff1.permute(0,2,3,1).cpu().detach().numpy()
        diff2=diff2.permute(0,2,3,1).cpu().detach().numpy()
        edge=edge.permute(0,2,3,1).cpu().detach().numpy()


        for j in range(0,100):
            image=np.zeros((x1.shape[1]*3,x1.shape[2]*2,3))
            image[0:x1.shape[1],0:x1.shape[2],:]=(x1[j,:,:,:]+1)/2
            image[0:x1.shape[1],x1.shape[2]:x1.shape[2]*2,:]=(x2[j,:,:,:]+1)/2
            diff_image=(diff1[j,:,:,:]+1)/2
            image[x1.shape[1]:x1.shape[1]*2,0:x1.shape[2],:]=diff_image
            image[x1.shape[1]:x1.shape[1]*2,x1.shape[2]:x1.shape[2]*2,:]=(edge[j,:,:,:]+1)/2

            image[x1.shape[1]*2:x1.shape[1]*3,0:x1.shape[2],:]=(x_new1[j,:,:,:]+1)/2
            image[x1.shape[1]*2:x1.shape[1]*3,x1.shape[2]:x1.shape[2]*2,:]=(x_new2[j,:,:,:]+1)/2
            image=image*255
            image=image.astype(np.int32)
            cv2.imwrite(os.path.join(file_save_path,"test_"+str(step*100+j)+".jpg"),image)
        break

@click.command()
@click.option('--batch_size',default=100,type=int, help="the batch size of the test")
@click.option('--dataset_name',default="face_point",type=click.Choice(["mnist","mnist_style","face_point"]),help="the string that defines the current dataset use")
@click.option('--model_name',default="GAN_face_point",type=click.Choice(["GAN_mnist","GAN_mnist_style","GAN_face_point"]),help="the string that  defines the current model use")
@click.option('--model_save_path',default="checkpoints/",type=str,help="the model save path")
@click.option('--file_save_path',default="test_output",type=str,help="the model save path")
def test_face(batch_size,dataset_name,model_name,model_save_path,file_save_path):
    is_cuda=False
    if(torch.cuda.is_available()):
        is_cuda=True
        print("cuda is available, current is cuda mode")
    else:
        print("cuda is unavailable, current is cpu mode")
    models=generate_models(model_name)
    train_eval_switch(models,False)

    if(is_cuda):
        for i in range(0,len(models)):
            models[i]=models[i].cuda()
    print("restoring models....")
    restore_models(models,model_name,dataset_name,model_save_path)
    print("restore models succeed")
    encoder=models[0]
    decoder=models[1]

    mnist_loader,edge_loader=generate_dataset(dataset_name,batch_size,train=False)


    for step,(x1,x2) in enumerate(mnist_loader):
        #print x1.size()
        edge=edge_loader.next()
        if(is_cuda):
            x1=x1.cuda()
            x2=x2.cuda()
            edge=edge.cuda()

        same1,diff1=encoder(x1)
        same2,diff2=encoder(x2)
        x_new1=decoder(same1,diff1)
        x_new2=decoder(same2,edge)
        x1=x1.permute(0,2,3,1).cpu().numpy()
        x2=x2.permute(0,2,3,1).cpu().numpy()
        x_new1=x_new1.permute(0,2,3,1).cpu().detach().numpy()
        x_new2=x_new2.permute(0,2,3,1).cpu().detach().numpy()
        diff1=diff1.permute(0,2,3,1).cpu().detach().numpy()
        diff2=diff2.permute(0,2,3,1).cpu().detach().numpy()
        edge=edge.permute(0,2,3,1).cpu().detach().numpy()


        for j in range(0,100):
            image=np.zeros((x1.shape[1]*3,x1.shape[2]*2,3))
            image[0:x1.shape[1],0:x1.shape[2],:]=(x1[j,:,:,:]+1)/2
            image[0:x1.shape[1],x1.shape[2]:x1.shape[2]*2,:]=(x2[j,:,:,:]+1)/2
            diff_image=(diff1[j,:,:,:]+1)/2
            image[x1.shape[1]:x1.shape[1]*2,0:x1.shape[2],:]=diff_image
            image[x1.shape[1]:x1.shape[1]*2,x1.shape[2]:x1.shape[2]*2,:]=(edge[j,:,:,:]+1)/2

            image[x1.shape[1]*2:x1.shape[1]*3,0:x1.shape[2],:]=(x_new1[j,:,:,:]+1)/2
            image[x1.shape[1]*2:x1.shape[1]*3,x1.shape[2]:x1.shape[2]*2,:]=(x_new2[j,:,:,:]+1)/2
            image=image*255
            image=image.astype(np.int32)
            cv2.imwrite(os.path.join(file_save_path,"test_"+str(step*100+j)+".jpg"),image)
        break


@click.group()
def main():
    pass

if __name__ == '__main__':
    main.add_command(train)
    main.add_command(test_color)
    main.add_command(test_edge)
    main.add_command(test_face)
    main()
