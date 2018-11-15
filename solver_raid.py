import torch
import torch.nn as nn
import torch.utils.data as Data
import os
import random
import cv2
import numpy as np
import click
import utils.evaluate as evl

from utils.common_tools import *
import dataset.mnist_color.mnist_type as mnist_type


os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def init_models(models,init_type="default"):
    init_func=weights_init(init_type)
    for model in models:
        model.apply(init_func)
        model.cuda()


def zero_grad_for_all(optimizers):
    for optimizer in optimizers:
        optimizer.zero_grad()


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

    file_name=["encoder.pkl","decoder.pkl","image_dis.pkl","feature_classifier.pkl","verifier.pkl"]
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
    #,"decoder.pkl","image_dis.pkl","feature_classifier.pkl","verifier.pkl"

    file_name=["encoder.pkl"]
    for i in range(0,1):
        models[i].load_state_dict(torch.load(os.path.join(path,file_name[i])))


def report_loss(reconst_loss,image_D_loss,image_G_loss,feature_D_loss,feature_G_loss,verification_loss,step):
    print("In step %d  rstl:%.4f idl:%.4f igl:%.4f fdl:%.4f fgl:%.4f vl:%.4f"%(step,reconst_loss.cpu().item(),image_D_loss.cpu().item(),
    image_G_loss.cpu().item(),feature_D_loss.cpu().item(),feature_G_loss.cpu().item(),verification_loss.cpu().item()))


def train_eval_switch(models,train=True):
    if(train):
        for model in models:
            model.train()
    else:
        for model in models:
            model.eval()


def extract_feature(encoder,data_loader):
    features_array=None
    label_array=None
    cam_array=None

    for step,(x1,label,cam) in enumerate(data_loader):
        x1=x1.cuda()
        label=label.cuda()
        label=label.view(-1)
        s,d=encoder(x1)
        if(features_array is None):
            features_array=s.detach()
        else:
            features_array=torch.cat((features_array,s.detach()),0)
        if(label_array is None):
            label_array=label
        else:
            label_array=torch.cat((label_array,label),0)
        if(cam_array is None):
            cam_array=cam
        else:
            cam_array=torch.cat((cam_array,cam),0)
    return features_array.view(features_array.size()[0],-1).numpy(),label_array.cpu().numpy(),cam_array.cpu().numpy()


def calculate_score(verifier,features1,label1,features2,label2):
    scores=np.zeros(10000)
    sample_number=np.zeros(10000)
    top1=0
    top5=0
    top10=0
    for i in range(0,features1.size()[0]):
        if(i%100==0):
            print("current/total: %d/%d"%(i,features1.size()[0]))
        feature=features1[i]
        current_label=label1[i]
        # print(features2)
        #print("current_label")
        # print(current_label)
        feature=feature.repeat(features2.size()[0],1)
        score=torch.sum((feature-features2).pow(2),1)
        score=score.detach().cpu().numpy()
        # score=score.detach().cpu().numpy()
        arg=np.argsort(score)
        #print(arg)
        score=score[arg]
        #print("score")
        #print(score)
        score=score[0:10]
        labels=label2
        labels=labels[arg]
        #print("test labels")
        #print(labels)
        labels=labels[0:10]
        #print("test:"+str(labels[0])+":"+str(labelname2[labels[0]])+"query:"+str(current_label)+":"+str(labelname1[current_label]))
        if(labels[0]==current_label):
            top1+=1
        if(current_label in labels[0:5]):
            top5+=1
        if(current_label in labels[0:10]):
            top10+=1

        current_find=0
        total_score=0.0
        for i in range(0,10):
            if(current_label==labels[i]):
                current_find+=1
                total_score+=float(current_find)/(i+1)
        avg_score=0
        if(current_find!=0):
            avg_score=total_score/current_find
        scores[int(current_label)]+=avg_score
        sample_number[int(current_label)]+=1
    scores=scores[sample_number>0]
    sample_number=sample_number[sample_number>0]
    for i in range(0,len(scores)):
        scores[i]=scores[i]/sample_number[i]
    mAP=np.mean(scores)
    top1_acc=float(top1)/features1.size()[0]
    top5_acc=float(top5)/features1.size()[0]
    top10_acc=float(top10)/features1.size()[0]

    return top1_acc,top5_acc,top10_acc,mAP

def test_data(encoder,verifier,data_loader1,data_loader2):
    features1,label1,cam1=extract_feature(encoder,data_loader1)
    features2,label2,cam2=extract_feature(encoder,data_loader2)
    # top1_acc,top5_acc,top10_acc,mAP=calculate_score(verifier,features1,label1,features2,label2)
    return evl.get_evaluate(features1,label1,cam1,features2,label2,cam2)
    # return top1_acc,top5_acc,top10_acc,mAP




@click.command()
@click.option('--batch_size',default=8,type=int, help="the batch size of train")
@click.option('--epoch',default=500,type=int, help="the total epoch of train")
@click.option('--dataset_name',default="DukeMTMC-reID",type=click.Choice(["mnist_type","DukeMTMC-reID"]),help="the string that defines the current dataset use")
@click.option('--model_name',default="GAN_Duke",type=click.Choice(["GAN_mnist","GAN_Duke"]),help="the string that  defines the current model use")
@click.option('--learning_rate',default=[0.01,0.1,0.001,0.001,0.1],nargs=5,type=float,help="the learning_rate of the four optimizer")
@click.option('--reconst_param',default=10.0,type=float,help="the reconstion loss coefficient")
@click.option('--image_g_loss_param',default=1.0,type=float,help="the image discriminator loss coefficient")
@click.option('--feature_g_loss_param',default=1.0,type=float,help="the feature discriminator loss coefficient")
@click.option('--model_save_path',default="checkpoints/",type=str,help="the model save path")
@click.option('--train_method',default="lsgan",type=click.Choice(["lsgan","wgan","hinge"]),help="the loss type of the train")
@click.option('--init_weight_type',default="orthogonal",type=click.Choice(["default","gaussian","xavier","kaiming","orthogonal"]),help="the loss type of the train")
@click.option('--optimizer_type',default="sgd",type=click.Choice(["adam","sgd"]),help="the loss type of the train")
@click.option('--verify_loss_param',default=1,type=float,help="the same constrain")
@click.option('--steps_per_tune',default=100,type=float,help="steps per tunes to swith the main work")
def train(batch_size,epoch,dataset_name,model_name,learning_rate,reconst_param,image_g_loss_param,feature_g_loss_param,model_save_path,train_method,init_weight_type,optimizer_type,verify_loss_param,steps_per_tune):

    models=generate_models(model_name)

    init_models(models,optimizer_type)

    optimizers=generate_optimizers(models,learning_rate,optimizer_type)

    data_loader=generate_dataset(dataset_name,batch_size,train=True)

    query_loader,test_loader=generate_dataset(dataset_name,batch_size,train=False)

    encoder_optimizer,decoder_optimizer,image_d_optimizer,feature_d_optimizer,verifier_optimizer=optimizers

    encoder,decoder,image_dis,feature_dis,verifier=models

    best_acc=0

    switch_epoch=[0,75,100]

    learning_rates=[
    [0.001,0.01,0.0001,0.0001,0.01],
    [0.0001,0.001,0.0001,0.0001,0.001],
    [0.00001,0.01,0.0001,0.0001,0.0001]
    ]

    for i in range(0,epoch):
        train_eval_switch(models)
        if(i%1==0):
            print("saving model....")
            save_models(models,model_name,dataset_name,model_save_path)
            print("save model succeed")
        print("begin the epoch : %d"%(i))
        tune=0

        #switch to new optimizer
        if(i in switch_epoch):
            ind=switch_epoch.index(i)
            optimizers=generate_optimizers(models,learning_rates[ind],optimizer_type)
            encoder_optimizer,decoder_optimizer,image_d_optimizer,feature_d_optimizer,verifier_optimizer=optimizers

        for step,(x1,x2,clas) in enumerate(data_loader):
            x1=x1.cuda()
            x2=x2.cuda()
            clas=clas.cuda()

            if(step%steps_per_tune==0):
                tune=1-tune
            #just optimize the feature discriminator
            if(tune==0 and step%10!=0):
                #encoder image
                s1,d1=encoder(x1)
                s2,d2=encoder(x2)
                #discriminator for feature
                d_f1=feature_dis(d1)
                d_f2=feature_dis(d2)
                #calculate the feature discriminator loss
                feature_d_loss=(D_classify_loss(d_f1,clas)+D_classify_loss(d_f2,clas))/2
                feature_d_loss.backward()
                feature_d_optimizer.step()
                zero_grad_for_all(optimizers)
            else:
                #print("222")
                #encoder image
                s1,d1=encoder(x1)
                s2,d2=encoder(x2)
                #decoder image and produce im
                x1_fake=decoder(s2,d1)
                x2_fake=decoder(s1,d2)
                #discriminator for feature
                d_f1=feature_dis(d1)
                d_f2=feature_dis(d2)
                #discriminator for image real and fake
                d_i1r=image_dis(x1)
                d_i2r=image_dis(x2)
                d_i1f=image_dis(x1_fake)
                d_i2f=image_dis(x2_fake)
                #calculate the verifier label
                label=torch.argmax(clas,1)
                label=torch.cat((label,label),0)
                feature=torch.cat((s1,s2),0)
                #calculate the verify result
                v_loss=verifier(feature,label)
                #calculate the real image and fake image l1 loss
                reconst_loss=reconstruction_loss(x1,x1_fake)+reconstruction_loss(x2,x2_fake)
                #calculate the image discriminator loss
                image_d_loss=(D_real_loss(d_i1r,train_method)+D_real_loss(d_i2r,train_method)+D_fake_loss(d_i1f,train_method)+D_fake_loss(d_i2f,train_method))/2
                #calculate the image generator loss
                image_g_loss=(G_fake_loss(d_i1f,train_method)+G_fake_loss(d_i2f,train_method))/2
                #calculate the feature discriminator loss
                feature_d_loss=(D_classify_loss(d_f1,clas)+D_classify_loss(d_f2,clas))/2
                #calculate the feature generator loss
                feature_g_loss=(G_classify_loss(d_f1)+G_classify_loss(d_f2))/2
                #calculate the verification loss
                #calcuate the total loss of the multitask
                total_loss=reconst_param*reconst_loss+image_g_loss_param*image_g_loss+feature_g_loss_param*feature_g_loss

                total_loss=verify_loss_param*v_loss

                # if(step%50==0):
                #     print(total_loss.detach().cpu().item())

                if(step%500==0):
                    report_loss(reconst_loss,image_d_loss,image_g_loss,feature_d_loss,feature_g_loss,v_loss,step)

                if(tune!=0):
                    #optimize for the discriminator
                    image_d_loss.backward(retain_graph=True)
                    image_d_optimizer.step()
                    zero_grad_for_all(optimizers)

                    #optimize for the encoder decoder and verifier
                    total_loss.backward()
                    verifier_optimizer.step()
                    decoder_optimizer.step()
                    encoder_optimizer.step()
                    zero_grad_for_all(optimizers)
                    #print(feature1)

        train_eval_switch(models,False)
        top1_acc,top5_acc,top10_acc,mAP=test_data(encoder,verifier,query_loader,test_loader)
        print("the top acc is: %.4f %.4f %.4f mAP: %.4f"%(top1_acc,top5_acc,top10_acc,mAP))
        if(top1_acc>best_acc):
            best_acc=top1_acc
            save_models(models,model_name+"_best",dataset_name,model_save_path)







@click.command()
@click.option('--batch_size',default=5,type=int, help="the batch size of the test")
@click.option('--dataset_name',default="DukeMTMC-reID",type=click.Choice(["mnist_type","DukeMTMC-reID"]),help="the string that defines the current dataset use")
@click.option('--model_name',default="GAN_Duke",type=click.Choice(["GAN_mnist","GAN_Duke"]),help="the string that  defines the current model use")
@click.option('--model_save_path',default="checkpoints/",type=str,help="the model save path")
@click.option('--file_save_path',default="test_output",type=str,help="the model save path")
def test(batch_size,dataset_name,model_name,model_save_path,file_save_path):

    models=generate_models(model_name)
    train_eval_switch(models,False)

    print("restoring models....")
    restore_models(models,model_name,dataset_name,model_save_path)
    print("restore models succeed")
    encoder=models[0]
    verifer=models[4]

    query_loader,test_loader=generate_dataset(dataset_name,batch_size,train=False)
    #print(labelname_test)
    #print(labelname_query)
    top1_acc,top5_acc,top10_acc,mAP=test_data(encoder,verifer,query_loader,test_loader)
    print("the top acc is: %.4f %.4f %.4f mAP: %.4f"%(top1_acc,top5_acc,top10_acc,mAP))



@click.group()
def main():
    pass

if __name__ == '__main__':
    main.add_command(train)
    main.add_command(test)
    main()
