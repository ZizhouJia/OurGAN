import torch
import torch.nn as nn
import torch.utils.data as Data
import os
import random
import cv2
import numpy as np
import click

from utils.common_tools import *
import dataset.mnist_color.mnist_type as mnist_type
import dataset.reid.reid_dataset as reid_dataset


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

    file_name=["encoder.pkl","decoder.pkl","image_dis.pkl","feature_classifier.pkl","verifier.pkl"]
    for i in range(0,len(models)):
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
    for step,(x1,label) in enumerate(data_loader):
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
    return features_array.view(features_array.size()[0],-1),label_array.cpu().numpy()


def calculate_score(verifier,features1,label1,features2,label2):
    scores=np.zeros(features1.size()[0])
    sample_number=np.zeros(features1.size()[0])
    total_correct=0
    for i in range(0,len(features1)):
        if(i%100==0):
            print("current/total: %d/%d"%(i,features1.size()[0]))
        feature=features1[i]
        current_label=label1[i]
        feature=feature.repeat(features2.size()[0],1)
        score=torch.sum((feature-features2).pow(2),1)
        score=score.detach().cpu().numpy()
        # score=score.detach().cpu().numpy()
        arg=np.argsort(score)
        score=score[arg]
        score=score[0:10]
        labels=label2
        labels=labels[arg]
        labels=labels[0:10]
        if(labels[0]==current_label):
            total_correct+=1
        current_find=0
        total_score=0
        for i in range(0,10):
            if(current_label==labels[i]):
                current_find+=1
                total_score+=current_find/(i+1)
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
    top1_acc=float(total_correct)/features1.size()[0]
    return top1_acc,mAP

def test_data(encoder,verifier,data_loader1,data_loader2):
    features1,label1=extract_feature(encoder,data_loader1)
    features2,label2=extract_feature(encoder,data_loader2)
    top1_acc,mAP=calculate_score(verifier,features1,label1,features2,label2)
    return top1_acc,mAP




@click.command()
@click.option('--batch_size',default=8,type=int, help="the batch size of train")
@click.option('--epoch',default=100,type=int, help="the total epoch of train")
@click.option('--dataset_name',default="mnist_type",type=click.Choice(["mnist_type"]),help="the string that defines the current dataset use")
@click.option('--model_name',default="GAN_mnist",type=click.Choice(["GAN_mnist"]),help="the string that  defines the current model use")
@click.option('--learning_rate',default=[0.001,0.001,0.001,0.001,0.01],nargs=5,type=float,help="the learning_rate of the four optimizer")
@click.option('--reconst_param',default=10.0,type=float,help="the reconstion loss coefficient")
@click.option('--image_g_loss_param',default=1.0,type=float,help="the image discriminator loss coefficient")
@click.option('--feature_g_loss_param',default=1.0,type=float,help="the feature discriminator loss coefficient")
@click.option('--model_save_path',default="checkpoints/",type=str,help="the model save path")
@click.option('--train_method',default="lsgan",type=click.Choice(["lsgan","wgan","hinge"]),help="the loss type of the train")
@click.option('--init_weight_type',default="xavier",type=click.Choice(["default","gaussian","xavier","kaiming","orthogonal"]),help="the loss type of the train")
@click.option('--optimizer_type',default="adam",type=click.Choice(["adam","sgd"]),help="the loss type of the train")
@click.option('--verify_loss_param',default=1,type=float,help="the same constrain")
@click.option('--steps_per_tune',default=100,type=float,help="steps per tunes to swith the main work")
def train(batch_size,epoch,dataset_name,model_name,learning_rate,reconst_param,image_g_loss_param,feature_g_loss_param,model_save_path,train_method,init_weight_type,optimizer_type,verify_loss_param,steps_per_tune):

    models=generate_models(model_name)

    init_models(models,optimizer_type)

    train_eval_switch(models)

    optimizers=generate_optimizers(models,learning_rate,optimizer_type)

    data_loader=generate_dataset(dataset_name,batch_size,train=True)

    encoder_optimizer,decoder_optimizer,image_d_optimizer,feature_d_optimizer,verifier_optimizer=optimizers

    encoder,decoder,image_dis,feature_dis,verifier=models


    for i in range(0,epoch):
        if(i%1==0):
            print("saving model....")
            save_models(models,model_name,dataset_name,model_save_path)
            print("save model succeed")

        print("begin the epoch : %d"%(i))
        tune=0
        for step,(x1,x2,clas) in enumerate(data_loader):
            x1=x1.cuda()
            x2=x2.cuda()
            clas=clas.cuda()

            if(step%steps_per_tune==0):
                tune=1-tune
            #tune=1
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
                #decoder image and produce image
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
                id=torch.argmax(clas,1)
                half_size=id.size()[0]/2
                label=(id[:half_size]==id[half_size:]).long()
                # label2=(id==id).long()
                feature1=torch.cat((s1[:half_size],s2[:half_size]),0)
                feature2=torch.cat((s1[half_size:],s2[half_size:]),0)
                label=torch.cat((label,label),0)
                #calculate the verify result
                v_pred=verifier(feature1,feature2)
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
                v_loss=verify_loss(v_pred,label)
                #calcuate the total loss of the multitask
                total_loss=reconst_param*reconst_loss+image_g_loss_param*image_g_loss+feature_g_loss_param*feature_g_loss
                total_loss+=verify_loss_param*v_loss

                if(step%100==0):
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
    top1_acc,mAP=test_data(encoder,verifer,query_loader,test_loader)
    print("the top1 acc is: %.4f mAP: %.4f"%(top1_acc,mAP))



@click.group()
def main():
    pass

if __name__ == '__main__':
    main.add_command(train)
    main.add_command(test)
    main()
