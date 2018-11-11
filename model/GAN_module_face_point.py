import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class encoder(nn.Module):
    def __init__(self):
        super(encoder,self).__init__()

        dim=[16,32,64,64]
        diff_dim=[32,16,1]

        if(not torch.cuda.is_available()):
            dim=[3,3,5,5]
            content_dim=[5,3,3]

        self.encoder_down1=down(3,dim[0])
        self.encoder_down2=down(dim[0],dim[1])
        self.encoder_down3=down(dim[1],dim[2])
        self.encoder_resblocks=res_blocks(2,dim[2])

        self.dropout=nn.Dropout(p=0.5)


        same=[]
        same.append(down(dim[2],dim[3]))
        same.append(nn.LeakyReLU(0.01))
        self.encoder_same=nn.Sequential(*same)


        self.encoder_diff1=convtr_single(dim[2],diff_dim[0])
        self.encoder_diff2=convtr_single(diff_dim[0],diff_dim[1])


        self.encoder_diff3=nn.ConvTranspose2d(diff_dim[1],diff_dim[2], kernel_size=4, stride=2, padding=1, bias=False)

        self.encoder_tanh=nn.Tanh()



    def forward(self,x):

        x1=self.encoder_down1(x)
        #print x1.size()
        x2=self.encoder_down2(x1)
        #print x2.size()
        x3=self.encoder_down3(x2)
        #print x3.size()
        x4=self.encoder_resblocks(x3)
        #print x4.size()
        out=x4

        media=[]
        media.append(x1)
        media.append(x2)
        media.append(x3)
        media.append(x4)


        #out=self.encoder_conv(x)
        same=self.encoder_same(out)
        media.append(same)


        diff=self.encoder_diff1(out)
        #print("diff1")
        #print(diff.size())
        diff=self.encoder_diff2(diff)
        #print("diff2")
        #print(diff.size())
        diff=self.encoder_diff3(diff)
        #print("diff3")
        #print(diff.size())
        diff=self.encoder_tanh(diff)
        #print("diff4")
        #print(diff.size())
        return media,diff


class decoder(nn.Module):
    def __init__(self):
        super(decoder,self).__init__()

        dim=[64,64,32,16]
        diff_dim=[32,16,3]

        if(not torch.cuda.is_available()):
            dim=[5,5,3,3]
            content_dim=[3,3,5]

        same=[]
        same.append(convtr_single(dim[0],dim[1]))
        same.append(nn.LeakyReLU(0.01))
        self.decoder_same=nn.Sequential(*same)

        diff=[]
        diff.append(conv_single(1,dim[3]))
        diff.append(conv_single(dim[3],dim[2]))
        diff.append(conv_single(dim[2],dim[1]))

        #diff.append(nn.LeakyReLU(0.01))
        #content.append(nn.InstanceNorm2d(1,affine=True,track_running_stats=True))
        diff.append(nn.Tanh())
        self.decoder_diff=nn.Sequential(*diff)

        self.two2one=conv_single(2*dim[0], dim[0], kernel_size=3, stride=1, padding=1)

        self.decoder_down1=up(dim[0]*2,diff_dim[0])
        self.decoder_down2=up(diff_dim[0]*2,diff_dim[1])

        self.decoder_down3=up(diff_dim[1]*2,diff_dim[2])
        self.decoder_resblocks=res_blocks(2,dim[0])

        self.decoder_tanh=nn.Tanh()

    def forward(self,media,diff):
        x1=media[0]
        x2=media[1]
        x3=media[2]
        x4=media[3]
        same=media[4]
        same=self.decoder_same(same)
        #print(diff.size())
        diff=self.decoder_diff(diff)
        out=torch.cat((same,diff),1)

        out=self.two2one(out)

        out=self.decoder_resblocks(out)
        out=self.decoder_down1(out,x3)
        out=self.decoder_down2(out,x2)
        out=self.decoder_down3(out,x1)
        out=self.decoder_tanh(out)

        #out=self.decoder_down1(out)
        #out=self.decoder_tanh(out)

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
            layers.append(nn.LeakyReLU(0.01))
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
            layers.append(nn.LeakyReLU(0.01))
        layers.append(nn.Conv2d(dim[3],1,kernel_size=4,stride=1,padding=0))
        self.dis=nn.Sequential(*layers)

    def forward(self,x):
        return self.dis(x)


class res_blocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(res_blocks, self).__init__()
        layers =[]
        for i in xrange(num_blocks):
            layers += [res_block(dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class res_block(nn.Module):
    def __init__(self, dim):
        super(res_block, self).__init__()

        model=[]
        model.append(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(nn.BatchNorm2d(dim))
        #model.append(nn.InstanceNorm2d(dim,affine=True,track_running_stats=True))
        model.append(nn.LeakyReLU(0.01))
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class convtr_single(nn.Module):
    def __init__(self, inputdim, outputdim, kernel_size=4, stride=2, padding=1):
        super(convtr_single,self).__init__()
        layers = []
        layers.append(nn.ConvTranspose2d(inputdim, outputdim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
        layers.append(nn.BatchNorm2d(outputdim))
        #layers.append(nn.InstanceNorm2d(outputdim, affine=True, track_running_stats=True))
        layers.append(nn.LeakyReLU(0.01))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class conv_single(nn.Module):
    def __init__(self, inputdim, outputdim, kernel_size=4, stride=2, padding=1):
        super(conv_single,self).__init__()
        layers = []
        layers.append(nn.Conv2d(inputdim, outputdim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
        layers.append(nn.BatchNorm2d(outputdim))
        #layers.append(nn.InstanceNorm2d(outputdim, affine=True, track_running_stats=True))
        layers.append(nn.LeakyReLU(0.01))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class down(nn.Module):
    def __init__(self,inputdim,outputdim,kernel_size=4,stride=2,padding=1):
        super(down,self).__init__()
        layers = []
        #layers.append(nn.MaxPool2d(2))
        layers.append(conv_single(inputdim,outputdim,kernel_size,stride,padding))
        self.model = nn.Sequential(*layers)

    def forward(self,x):
        x=self.model(x)
        return x


class up(nn.Module):
    def __init__(self, inputdim, outputdim,kernel_size=4,stride=2,padding=1, bilinear=True):
        super(up,self).__init__()
        self.up=nn.ConvTranspose2d(inputdim,outputdim,kernel_size=kernel_size,padding=padding,stride=stride)
        self.bn=nn.BatchNorm2d(outputdim)
        self.outputdim=outputdim


    def forward(self,x1,x2):

        x=torch.cat((x2,x1),1)
        x=self.up(x)
        if self.outputdim!=3:
            x=self.bn(x)

        return x
