import os
import torch
import torch.nn as nn

class encoder(nn.Module):
    def __init__(self):
        super(encoder,self).__init__()

        dim=[16,32,33]

        layers=[]
        layers.append(nn.Conv2d(3,dim[0],kernel_size=5,stride=1,padding=2,bias=False))
        layers.append(nn.InstanceNorm2d(dim[0],affine=True,track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        for i in range(2):
            layers.append(nn.Conv2d(dim[i],dim[i+1],kernel_size=4,stride=2,padding=1,bias=False))
            layers.append(nn.InstanceNorm2d(dim[i+1],affine=True,track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))

        self.encoder_conv=nn.Sequential(*layers)

        difference=[]
        difference.append(nn.Conv2d(1,10,kernel_size=7,stride=1,padding=0,bias=False))
        difference.append(nn.InstanceNorm2d(10,affine=True,track_running_stats=True))
        difference.append(nn.ReLU(inplace=True))
        self.encoder_diff=nn.Sequential(*difference)

    def forward(self,x):
        out=self.encoder_conv(x)
        diff=self.encoder_diff(out[:,-2:-1,:,:])
        return out[:,:-2,:,:],diff


class decoder(nn.Module):
    def __init__(self):
        super(decoder,self).__init__()

        dim=[33,32,16]
        difference=[]
        difference.append(nn.ConvTranspose2d(1,10,kernel_size=7,stride=1,padding=0,bias=True))
        difference.append(nn.InstanceNorm2d(1,affine=True,track_running_stats=True))
        difference.append(nn.ReLU())
        self.decoder_diff=nn.Sequential(*difference)

        layers=[]
        for i in range(2):
            layers.append(nn.ConvTranspose2d(dim[i],dim[i+1],kernel_size=4,stride=2,padding=1,bias=False))
            layers.append(nn.InstanceNorm2d(dim[i+1],affine=True,track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(dim[2],3,kernel_size=5,stride=1,padding=2,bias=False))
        layers.append(nn.Tanh())

        self.decoder_conv=nn.Sequential(*layers)

    def forward(self,conv,diff):
        diff_conv=self.decoder_diff(diff)
        out=torch.cat((conv,diff_conv),dimension=1)
        out=self.decoder_conv(out)
        return out

class discriminator_for_image(nn.Module):
    def __init__(self):
        super(discriminator_for_image,self).__init__()

        dim=[3,16,32,64]
        layers=[]

        for i in range(0,3):
            layers.append(nn.Conv2d(dim[i],dim[i+1],kernel_size=3,stride=2,padding=1,bias=False))
            layers.append(nn.ReLU())

        layers.append(nn.Conv2d(dim[3],1,kernel_size=4,stride=,padding=0))

        self.dis=nn.Sequential(*layers)

    def forward(self,x):
        return self.dis(x)


class discriminator_for_difference(nn.Module):
    def __init__(self):
        super(discriminator_for_difference,self).__init__()

        layers=[]
        layers.append(nn.Conv2d(10,1,kernel_size=1,stride=1,padding=0))
        layers.append(nn.Tanh())

        self.dis=nn.Sequential(*layers)

    def forward(self,x):
        return self.dis(x)
