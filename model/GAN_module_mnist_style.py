import os
import torch
import torch.nn as nn

class encoder(nn.Module):
    def __init__(self):
        super(encoder,self).__init__()

        dim=[16,32,32,32]
        content_dim=[32,16,8]

        if(not torch.cuda.is_available()):
            dim=[3,3,5,5]
            content_dim=[5,3,3]

        layers=[]
        layers.append(nn.Conv2d(3,dim[0],kernel_size=5,stride=1,padding=2,bias=False))
        layers.append(nn.InstanceNorm2d(dim[0],affine=True,track_running_stats=True))
        layers.append(nn.LeakyReLU(0.01,inplace=True))

        for i in range(2):
            layers.append(nn.Conv2d(dim[i],dim[i+1],kernel_size=4,stride=2,padding=1,bias=False))
            layers.append(nn.InstanceNorm2d(dim[i+1],affine=True,track_running_stats=True))
            layers.append(nn.LeakyReLU(0.01,inplace=True))
        self.encoder_conv=nn.Sequential(*layers)

        style=[]
        style.append(nn.Conv2d(dim[2],dim[3],kernel_size=4,stride=2,padding=1,bias=False))
        style.append(nn.InstanceNorm2d(dim[3],affine=True,track_running_stats=True))
        style.append(nn.LeakyReLU(0.01,inplace=True))
        style.append(nn.Conv2d(dim[3],32,kernel_size=3,stride=1,padding=0,bias=True))
        # style.append(nn.InstanceNorm2d(10,affine=True,track_running_stats=True))
        style.append(nn.LeakyReLU(0.01,inplace=True))
        self.encoder_style=nn.Sequential(*style)

        content=[]
        for i in range(2):
            content.append(nn.ConvTranspose2d(content_dim[i],content_dim[i+1],kernel_size=4,stride=2,padding=1,bias=False))
            layers.append(nn.InstanceNorm2d(content_dim[i+1],affine=True,track_running_stats=True))
            layers.append(nn.LeakyReLU(0.01,inplace=True))
        content.append(nn.Conv2d(content_dim[2],1,kernel_size=3,stride=1,padding=1,bias=False))
        # content.append(nn.InstanceNorm2d(1,affine=True,track_running_stats=True))
        content.append(nn.Tanh())
        self.encoder_content=nn.Sequential(*content)


    def forward(self,x):
        out=self.encoder_conv(x)
        same=self.encoder_style(out)
        diff=self.encoder_content(out)
        return same,diff


class decoder(nn.Module):
    def __init__(self):
        super(decoder,self).__init__()

        dim=[32,32,32,16]
        content_dim=[8,16,32]

        if(not torch.cuda.is_available()):
            dim=[5,5,3,3]
            content_dim=[3,3,5]

        style=[]
        style.append(nn.ConvTranspose2d(32,dim[0],kernel_size=4,stride=1,padding=0,bias=False))
        style.append(nn.InstanceNorm2d(dim[0],affine=True,track_running_stats=True))
        style.append(nn.LeakyReLU(0.01,inplace=True))
        style.append(nn.ConvTranspose2d(dim[0],dim[1],kernel_size=3,stride=2,padding=1,bias=False))
        style.append(nn.InstanceNorm2d(dim[1],affine=True,track_running_stats=True))
        style.append(nn.LeakyReLU(0.01,inplace=True))
        self.decoder_style=nn.Sequential(*style)

        content=[]
        content.append(nn.Conv2d(1,content_dim[0],kernel_size=3,stride=1,padding=1,bias=False))
        content.append(nn.InstanceNorm2d(content_dim[0],affine=True,track_running_stats=True))
        content.append(nn.ReLU())

        for i in range(2):
            content.append(nn.Conv2d(content_dim[i],content_dim[i+1],kernel_size=4,stride=2,padding=1,bias=False))
            content.append(nn.InstanceNorm2d(content_dim[i+1],affine=True,track_running_stats=True))
            content.append(nn.LeakyReLU(0.01,inplace=True))
        self.decoder_content=nn.Sequential(*content)

        layers=[]
        dim[1]+=content_dim[2]
        for i in range(1,3):
            layers.append(nn.ConvTranspose2d(dim[i],dim[i+1],kernel_size=4,stride=2,padding=1,bias=False))
            layers.append(nn.InstanceNorm2d(dim[i+1],affine=True,track_running_stats=True))
            layers.append(nn.LeakyReLU(0.01,inplace=True))
        layers.append(nn.Conv2d(dim[3],3,kernel_size=5,stride=1,padding=2,bias=False))
        layers.append(nn.Tanh())
        self.decoder_conv=nn.Sequential(*layers)

    def forward(self,same,diff):
        same=self.decoder_style(same)
        diff=self.decoder_content(diff)
        out=torch.cat((same,diff),1)
        out=self.decoder_conv(out)
        return out

class discriminator_for_image(nn.Module):
    def __init__(self):
        super(discriminator_for_image,self).__init__()
        dim=[3,16,32,64]
        if(not torch.cuda.is_available()):
            dim=[3,3,3,4]
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
