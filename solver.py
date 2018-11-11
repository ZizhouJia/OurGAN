# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.utils.data as Data
import dataset.mnist_color.mnist as mnist

from utils.common_tools import *
import os
import cv2
import numpy as np
import click

def forward_and_get_loss(models,x1,x2,feature_real,step,train_method,classes=False,interpolation=0,same_constrain=False):
    encoder,decoder,dis_image,dis_feature=models
    media1,diff1=encoder(x1)
    media2,diff2=encoder(x2)
    x1_fake=decoder(media2,diff1)
    x2_fake=decoder(media1,diff2)
    x1_rec=decoder(media1,diff1)
    x2_rec=decoder(media2,diff2)
    dis_x1_real=dis_image(x1)
    dis_x2_real=dis_image(x2)
    dis_x1_fake=dis_image(x1_fake)
    dis_x2_fake=dis_image(x2_fake)

    dis_feature_fake_x1=dis_feature(diff1)
    dis_feature_fake_x2=dis_feature(diff2)
    reconst_loss=reconstruction_loss(x1,x1_fake)+reconstruction_loss(x2,x2_fake)+reconstruction_loss(x1,x1_rec)+reconstruction_loss(x2,x2_rec)
    D_image_loss=(D_real_loss(dis_x1_real,train_method)+D_real_loss(dis_x2_real,train_method))/2
    D_image_fake_loss=D_fake_loss(dis_x1_fake,train_method)+D_fake_loss(dis_x2_fake,train_method)
    G_image_loss=G_fake_loss(dis_x1_fake,train_method)+G_fake_loss(dis_x2_fake,train_method)
    D_feature_loss=0
    G_feature_loss=0
    if(not classes):
        dis_feature_real=dis_feature(feature_real)
        D_feature_loss=D_real_loss(dis_feature_real,train_method)+(D_fake_loss(dis_feature_fake_x1,train_method)+D_fake_loss(dis_feature_fake_x2,train_method))/2
        G_feature_loss=(G_fake_loss(dis_feature_fake_x1,train_method)+G_fake_loss(dis_feature_fake_x2,train_method))/2
    else:
        D_feature_loss=(D_classify_loss(dis_feature_fake_x1,feature_real)+D_classify_loss(dis_feature_fake_x2,feature_real))/2
        G_feature_loss=(G_classify_loss(dis_feature_fake_x1,feature_real)+G_classify_loss(dis_feature_fake_x2,feature_real))/2

    if(interpolation>0):
        for i in range(0,interpolation):
            inter=torch.rand(x1.size()[0],1,1,1).cuda()
            inter=diff1*inter+diff2*(1-inter)
            x_inter_fake=decoder(media1,inter)
            dis_x_inter_fake=dis_image(x_inter_fake)
            G_image_loss+=G_fake_loss(dis_x_inter_fake)
            D_image_fake_loss+=D_fake_loss(dis_x_inter_fake)
    G_image_loss=G_image_loss/(interpolation+2)
    D_image_loss+=(D_image_fake_loss)/(interpolation+2)

    if(not same_constrain):
        return reconst_loss,feature_D_loss,image_D_loss,G_image_loss,G_feature_loss
    else:
        same_constrain=feature_same_loss(media1,media2)

        return reconst_loss,D_feature_loss,D_image_loss,G_image_loss,G_feature_loss,same_constrain


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

def report_loss2(reconst_loss,feature_D_loss,image_D_loss,G_image_loss,G_feature_loss,same_constrain,step):
    print("In step %d  rstl:%.4f dfl:%.4f dil:%.4f gil:%.4f gfl:%.4f scl:%.4f"%(step,reconst_loss.cpu().item(),feature_D_loss.cpu().item(),
    image_D_loss.cpu().item(),G_image_loss.cpu().item(),G_feature_loss.cpu().item(),same_constrain.cpu().item()))


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
@click.option('--dataset_name',default="mnist_style",type=click.Choice(["mnist","mnist_style","face_point","mnist_type"]),help="the string that defines the current dataset use")
@click.option('--model_name',default="GAN_mnist_style",type=click.Choice(["GAN_mnist","GAN_mnist_style","GAN_face_point"]),help="the string that  defines the current model use")
@click.option('--learning_rate',default=[0.0001,0.0001,0.0001,0.01],nargs=4,type=float,help="the learning_rate of the four optimizer")
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
@click.option('--batch_size',default=32,type=int, help="the batch size of train")
@click.option('--epoch',default=100,type=int, help="the total epoch of train")
@click.option('--dataset_name',default="mnist_type",type=click.Choice(["mnist","mnist_style","face_point","mnist_type"]),help="the string that defines the current dataset use")
@click.option('--model_name',default="GAN_mnist",type=click.Choice(["GAN_mnist","GAN_mnist_style","GAN_face_point"]),help="the string that  defines the current model use")
@click.option('--learning_rate',default=[0.001,0.001,0.0001,0.001,0.001],nargs=5,type=float,help="the learning_rate of the four optimizer")
@click.option('--reconst_param',default=10.0,type=float,help="the reconstion loss coefficient")
@click.option('--image_d_loss_param',default=1.0,type=float,help="the image discriminator loss coefficient")
@click.option('--feature_d_loss_param',default=1.0,type=float,help="the feature discriminator loss coefficient")
@click.option('--model_save_path',default="checkpoints/",type=str,help="the model save path")
@click.option('--train_method',default="lsgan",type=click.Choice(["lsgan","wgan","hinge"]),help="the loss type of the train")
@click.option('--init_weight_type',default="xavier",type=click.Choice(["default","gaussian","xavier","kaiming","orthogonal"]),help="the loss type of the train")
@click.option('--optimizer_type',default="adam",type=click.Choice(["adam","sgd"]),help="the loss type of the train")
@click.option('--interplot',default=0,type=int,help="the interplot")
@click.option('--same_constrain_param',default=1,type=float,help="the same constrain")
def train_type(batch_size,epoch,dataset_name,model_name,learning_rate,reconst_param,image_d_loss_param,feature_d_loss_param,model_save_path,train_method,init_weight_type,optimizer_type,interplot,same_constrain_param):
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
    encoder_optimizer,decoder_optimizer,image_D_optimizer,feature_D_optimizer=optimizers
    for i in range(0,epoch):
        if(i%1==0):
            print("saving model....")
            save_models(models,model_name,dataset_name,model_save_path)
            print("save model succeed")

        print("begin the epoch : %d"%(i))
        for step,(x1,x2,clas) in enumerate(mnist_loader):
            if(is_cuda):
                x1=x1.cuda()
                x2=x2.cuda()
                clas=clas.cuda()
            #开始训练过程
            #先更新image discriminator
            reconst_loss,feature_D_loss,image_D_loss,G_image_loss,G_feature_loss,same_constrain=forward_and_get_loss(models,x1,x2,clas,step,train_method,True,interplot,True)
            image_D_loss.backward(retain_graph=True)
            image_D_optimizer.step()
            zero_grad_for_all(optimizers)

            #再更新feature discriminator
            if(step%20>=0 and step%10<10):
                feature_D_loss.backward(retain_graph=True)
                feature_D_optimizer.step()
                zero_grad_for_all(optimizers)

            #最后更新Generator
            if(step%20>=10 and step%10<20):
                total_loss=reconst_loss*reconst_param+G_image_loss*image_d_loss_param+same_constrain*same_constrain_param
                total_loss+=G_feature_loss*feature_d_loss_param
                total_loss.backward()
                decoder_optimizer.step()
                encoder_optimizer.step()
                zero_grad_for_all(optimizers)


            if(step%100==0):
                report_loss2(reconst_loss,feature_D_loss,image_D_loss,G_image_loss,G_feature_loss,same_constrain,step)


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

    mnist_loader=generate_dataset(dataset_name,batch_size,train=False)



    for step,(x1,x2) in enumerate(mnist_loader):
        noise1=noise_loader.next()
        noise2=noise_loader.next()
        if(is_cuda):
            x1=x1.cuda()
            x2=x2.cuda()
            noise1=noise1.cuda()
            noise2=noise2.cuda()

        same1,diff1=encoder(x1)
        same2,diff2=encoder(x2)
        x_new1=decoder(same1,diff2)
        x_new2=decoder(same2,diff1)
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


    imgfea_loader=generate_dataset(dataset_name,batch_size,train=False)


    for step,(x1,x2,edge) in enumerate(imgfea_loader):
        #print x1.size()
        #edge=edge_loader.next()
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

            image=image.astype(np.uint8)
            image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            #image.save(os.path.join(file_save_path,"test_"+str(step*100+j)+".jpg"))
            cv2.imwrite(os.path.join(file_save_path,"test_"+str(step*100+j)+".jpg"),image)
        break

@click.command()
@click.option('--batch_size',default=100,type=int, help="the batch size of the test")
@click.option('--dataset_name',default="mnist_type",type=click.Choice(["mnist","mnist_style","face_point","mnist_type"]),help="the string that defines the current dataset use")
@click.option('--model_name',default="GAN_mnist",type=click.Choice(["GAN_mnist","GAN_mnist_style","GAN_face_point"]),help="the string that  defines the current model use")
@click.option('--model_save_path',default="checkpoints/",type=str,help="the model save path")
@click.option('--file_save_path',default="test_output",type=str,help="the model save path")
@click.option('--test_cross_class',default=False,type=bool,help="the model save path")
def test_mnist_step(batch_size,dataset_name,model_name,model_save_path,file_save_path,test_cross_class):
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

    mnist_loader,noise_loader=generate_dataset(dataset_name,batch_size,False,test_cross_class)


    for step,(x1,x2,cls) in enumerate(mnist_loader):
        if(is_cuda):
            x1=x1.cuda()
            x2=x2.cuda()

        same1,diff1=encoder(x1)
        same2,diff2=encoder(x2)
        vector=diff2-diff1
        images=[]
        images.append(x1.permute(0,2,3,1).cpu().detach().numpy())
        for i in range(0,11):
            x_new=decoder(same1,diff1+vector*i/10)
            images.append(x_new.permute(0,2,3,1).cpu().detach().numpy())
        images.append(x2.permute(0,2,3,1).cpu().detach().numpy())

        for j in range(0,100):
            image=np.zeros((x1.shape[2]*1,x1.shape[3]*13,3))
            for i in range(0,13):
                image[0:x1.shape[2],x1.shape[3]*i:x1.shape[3]*(i+1),:]=(images[i][j,:,:,:]+1)/2
            image=image*255
            image=image.astype(np.int32)
            cv2.imwrite(os.path.join(file_save_path,"test_"+str(step*100+j)+".jpg"),image)
        break

@click.command()
@click.option('--batch_size',default=100,type=int, help="the batch size of the test")
@click.option('--dataset_name',default="mnist_type",type=click.Choice(["mnist","mnist_style","face_point","mnist_type"]),help="the string that defines the current dataset use")
@click.option('--model_name',default="GAN_mnist",type=click.Choice(["GAN_mnist","GAN_mnist_style","GAN_face_point"]),help="the string that  defines the current model use")
@click.option('--model_save_path',default="checkpoints/",type=str,help="the model save path")
@click.option('--file_save_path',default="test_output",type=str,help="the model save path")
def test_classify(batch_size,dataset_name,model_name,model_save_path,file_save_path):
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
    class_center=np.zeros((10,32))
    for c_number in range(0,10):
        dataset=mnist.minst(path="dataset/mnist_color/data/raw/",class_number=c_number,train=True)
        data_count=dataset.__len__()
        mnist_loader=Data.DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=0)
        for step, x in enumerate(mnist_loader):
            if(is_cuda):
                x=x.cuda()
            same,diff=encoder(x)
            class_center[c_number]+=(torch.sum(same.view(same.size()[0],-1),0)).detach().cpu().numpy()
        class_center[c_number]=class_center[c_number]/data_count
    total=0
    for c_number in range(0,10):
        dataset=mnist.minst(path="dataset/mnist_color/data/raw/",class_number=c_number,train=False)
        data_count=dataset.__len__()
        mnist_loader=Data.DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=0)
        for step, x in enumerate(mnist_loader):
            if(is_cuda):
                x=x.cuda()
            same,diff=encoder(x)
            same=same.view(same.size()[0],-1).detach().cpu().numpy()
            for i in range(0,len(same)):
                min_index=0
                distance=100.0
                for j in range(0,len(class_center)):
                    new_distance=np.mean(np.abs(class_center[j]-same[i]))
                    if(new_distance<distance):
                        min_index=j
                        distance=new_distance
                if(min_index==c_number):
                    total+=1
    print("the total acc is:%.4f"%(float(total)/10000))


@click.command()
@click.option('--batch_size',default=2,type=int, help="the batch size of train")
@click.option('--epoch',default=1000,type=int, help="the total epoch of train")
@click.option('--dataset_name',default="face_point",type=click.Choice(["mnist","mnist_style","face_point"]),help="the string that defines the current dataset use")
@click.option('--model_name',default="GAN_face_point",type=click.Choice(["GAN_mnist","GAN_mnist_style","GAN_face_point"]),help="the string that  defines the current model use")
@click.option('--learning_rate',default=[0.0001,0.0001,0.0001,0.0001],nargs=4,type=float,help="the learning_rate of the four optimizer")
@click.option('--reconst_param',default=10.0,type=float,help="the reconstion loss coefficient")
@click.option('--image_d_loss_param',default=1.0,type=float,help="the image discriminator loss coefficient")
@click.option('--feature_d_loss_param',default=1.0,type=float,help="the feature discriminator loss coefficient")
@click.option('--model_save_path',default="checkpoints/",type=str,help="the model save path")
@click.option('--train_method',default="lsgan",type=click.Choice(["lsgan","wgan","hinge"]),help="the loss type of the train")
@click.option('--init_weight_type',default="default",type=click.Choice(["default","gaussian","xavier","kaiming","orthogonal"]),help="the loss type of the train")
@click.option('--optimizer_type',default="adam",type=click.Choice(["adam","sgd"]),help="the loss type of the train")
def train_face(batch_size,epoch,dataset_name,model_name,learning_rate,reconst_param,image_d_loss_param,feature_d_loss_param,model_save_path,train_method,init_weight_type,optimizer_type):
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
    data_loader=generate_dataset(dataset_name,batch_size,train=True)

    #epoch的次数，这里先写成变量，以后会加入到config文件中
    encoder_optimizer,decoder_optimizer,image_D_optimizer,feature_D_optimizer=optimizers
    for i in range(0,epoch):
        if(i%1==0):
            print("saving model....")
            save_models(models,model_name,dataset_name,model_save_path)
            print("save model succeed")

        print("begin the epoch : %d"%(i))
        for step,(x1,x2,noise) in enumerate(data_loader):
            #print x1.size()
            #noise=noise_loader.next()
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



@click.group()
def main():
    pass

if __name__ == '__main__':
    main.add_command(train)
    main.add_command(train_type)
    main.add_command(test_color)
    main.add_command(test_edge)
    main.add_command(test_face)
    main.add_command(train_face)
    main.add_command(test_mnist_step)
    main.add_command(test_classify)
    main()
