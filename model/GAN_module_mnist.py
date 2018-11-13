import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class encoder(nn.Module):
    def __init__(self):
        super(encoder,self).__init__()

        layers=[]
        layers.append(nn.Conv2d(1,16,kernel_size=4,stride=1,padding=2,bias=False))
        layers.append(nn.BatchNorm2d(16))
        layers.append(nn.LeakyReLU(0.01,inplace=True))
        layers.append(nn.Conv2d(16,32,kernel_size=4,stride=2,padding=1,bias=False))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.LeakyReLU(0.01,inplace=True))
        layers.append(nn.Conv2d(32,64,kernel_size=4,stride=2,padding=1,bias=False))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.LeakyReLU(0.01,inplace=True))
        layers.append(nn.Conv2d(64,64,kernel_size=7,stride=1,padding=0,bias=True))
        layers.append(nn.BatchNorm2d(64))
        self.encoder_conv=nn.Sequential(*layers)


    def forward(self,x):
        out=self.encoder_conv(x)
        out_diff=out[:,-32:,:,:]
        out_conv=out[:,:-32,:,:]

        return out_conv,out_diff


class decoder(nn.Module):
    def __init__(self):
        super(decoder,self).__init__()

        layers=[]
        layers.append(nn.ConvTranspose2d(64,64,kernel_size=7,stride=1,padding=0,bias=False))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.LeakyReLU(0.01,inplace=True))
        layers.append(nn.ConvTranspose2d(64,32,kernel_size=4,stride=2,padding=1,bias=False))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.LeakyReLU(0.01,inplace=True))
        layers.append(nn.ConvTranspose2d(32,16,kernel_size=4,stride=2,padding=1,bias=False))
        layers.append(nn.BatchNorm2d(16))
        layers.append(nn.LeakyReLU(0.01,inplace=True))
        layers.append(nn.Conv2d(16,1,kernel_size=5,stride=1,padding=2,bias=False))
        layers.append(nn.Tanh())

        self.decoder_conv=nn.Sequential(*layers)

    def forward(self,conv,diff):
        out=torch.cat((conv,diff),1)
        out=self.decoder_conv(out)
        return out

class discriminator_for_image(nn.Module):
    def __init__(self):
        super(discriminator_for_image,self).__init__()
        dim=[1,16,32,64]
        if(not torch.cuda.is_available()):
            dim=[1,3,3,4]
        layers=[]

        for i in range(0,3):
            layers.append(nn.Conv2d(dim[i],dim[i+1],kernel_size=3,stride=2,padding=1,bias=False))
            layers.append(nn.LeakyReLU(0.01,inplace=True))

        layers.append(nn.Conv2d(dim[3],1,kernel_size=4,stride=1,padding=0))

        self.dis=nn.Sequential(*layers)

    def forward(self,x):
        return self.dis(x)


class discriminator_for_difference(nn.Module):
    def __init__(self):
        super(discriminator_for_difference,self).__init__()

        layers=[]
        layers.append(nn.Conv2d(32,32,kernel_size=1,stride=1,padding=0))
        layers.append(nn.LeakyReLU(0.01,inplace=True))
        layers.append(nn.Conv2d(32,32,kernel_size=1,stride=1,padding=0))
        layers.append(nn.LeakyReLU(0.01,inplace=True))
        layers.append(nn.Conv2d(32,10,kernel_size=1,stride=1,padding=0))
        self.dis=nn.Sequential(*layers)

    def forward(self,x):
        out=self.dis(x)
        return out.view(out.size()[0],-1)


class verifier(nn.Module):
    def __init__(self):
        super(verifier,self).__init__()
        self.bn=nn.BatchNorm2d(32)
        self.linear=nn.Conv2d(32,1,kernel_size=1,stride=1,padding=0)
        self.sig=nn.Sigmoid()

    def forward(self,feature1,feature2):
        feature1=feature1.view(-1,32,1,1)
        feature2=feature2.view(-1,32,1,1)
        feature=feature1-feature2
        feature=feature.pow(2)
        feature=self.bn(feature)
        feature=self.linear(feature)
        score=self.sig(feature)
        return score.view(-1)
