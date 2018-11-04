import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class encoder(nn.Module):
    def __init__(self):
        super(encoder,self).__init__()

        dim=[16,32,64,128]
        content_dim=[32,16,8]

        if(not torch.cuda.is_available()):
            dim=[3,3,5,5]
            content_dim=[5,3,3]

        self.encoder_conv1=inconv(3,dim[0])
        self.encoder_down1=down(dim[0],dim[1])
        self.encoder_down2=down(dim[1],dim[2])
        self.encoder_down3=down(dim[2],dim[3])
        self.encoder_down4=down(dim[3],dim[3])


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
        #content.append(nn.InstanceNorm2d(1,affine=True,track_running_stats=True))
        content.append(nn.Tanh())
        self.encoder_content=nn.Sequential(*content)


    def forward(self,x):

        x1=self.encoder_conv1(x)
        x2=self.encoder_down1(x1)
        x3=self.encoder_down2(x2)
        x4=self.encoder_down3(x3)
        x5=self.encoder_down4(x4)

        media=[]
        media.append(x1)
        media.append(x2)
        media.append(x3)
        media.append(x4)
        media.append(x5)

        #out=self.encoder_conv(x)
        same=self.encoder_style(x5)
        diff=self.encoder_content(x5)

        media.append(same)
        return media,diff


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


class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        layers =[]
        for i in xrange(num_blocks):
            layers += [ResBlocks(dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()

        model=[]
        model.append(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(nn.InstanceNorm2d(dim,affine=True,track_running_stats=True))
        model.append(nn.LeakyReLU(0.01,inplace=True))
        model.append(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(nn.InstanceNorm2d(dim,affine=True,track_running_stats=True))
        #layers.append(nn.LeakyReLU(0.01,inplace=True))
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class Conv_double(nn.Module):
    def __init__(self, inputdim, outputdim, kernel_size=3, stride=1, padding=1):
        layers = []
        layers.append(nn.Conv2d(inputdim, outputdim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
        layers.append(nn.InstanceNorm2d(outputdim, affine=True, track_running_stats=True))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        layers.append(nn.Conv2d(inputdim, outputdim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
        layers.append(nn.InstanceNorm2d(outputdim, affine=True, track_running_stats=True))
        layers.append(nn.LeakyReLU(0.01, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Conv_single(nn.Module):
    def __init__(self, inputdim, outputdim, kernel_size=3, stride=1, padding=1):
        layers = []
        layers.append(nn.Conv2d(inputdim, outputdim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
        layers.append(nn.InstanceNorm2d(outputdim, affine=True, track_running_stats=True))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class inconv(nn.Module):
    def __init__(self,inputdim,outputdim,kernel_size=3,stride=1,padding=0):
        super(inconv,self).__init__()
        self.conv = Conv_double(inputdim,outputdim,kernel_size,stride,padding)

    def forward(self,x):
        x=self.conv(x)
        return x

class down(nn.Module):
    def __init__(self,inputdim,outputdim,kernel_size=3,stride=1,padding=1):
        super(down,self).__init__()
        layers = []
        layers.append(nn.MaxPool2d(2))
        layers.append(Conv_double(inputdim,outputdim,kernel_size,stride,padding))
        self.model = nn.Sequential(*layers)

    def forward(self,x):
        x=self.model(x)
        return x


class up(nn.Module):
    def __init__(self, inputdim, outputdim,kernel_size=3,stride=2,padding=1, bilinear=True):
        if bilinear:
            self.up=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        else:
            self.up=nn.ConvTranspose2d(inputdim//2,inputdim//2,2,stride=2)
        self.conv=Conv_double(inputdim,outputdim,kernel_size,stride,padding)


    def forward(self,x1,x2):
        x1=self.up(x1)
        diffX=x1.size()[2]-x2.size()[2]
        diffY=x1.size()[3]-x2.size()[3]
        x2=F.pad(x2,(diffX//2,int(diffX/2)),diffY//2,int(diffY/2)))
        x=torch.cat([x2,x1],dim=1)
        x=self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self,inputdim,outputdim,kernel_size=1,stride=1,padding=0):
        super(outconv,self).__init__()
        self.conv=nn.Conv2d(inputdim,outputdim,kernel_size=kernel_size,stride=stride,padding=padding)

    def forward(self,x):
        x=self.conv(x)
        return x
