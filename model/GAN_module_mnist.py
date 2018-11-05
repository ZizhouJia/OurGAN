import os
import torch
import torch.nn as nn

class encoder(nn.Module):
    def __init__(self):
        super(encoder,self).__init__()

        dim=[16,32,36]

        if(not torch.cuda.is_available()):
            dim=[3,3,5]

        layers=[]
        layers.append(nn.Conv2d(3,dim[0],kernel_size=5,stride=1,padding=2,bias=False))
        layers.append(nn.InstanceNorm2d(dim[0],affine=True,track_running_stats=True))
        layers.append(nn.LeakyReLU(0.01,inplace=True))

        for i in range(2):
            # layers.append(nn.Conv2d(dim[i],dim[i],kernel_size=3,stride=1,padding=1,bias=False))
            # layers.append(nn.InstanceNorm2d(dim[i],affine=True,track_running_stats=True))
            # layers.append(nn.LeakyReLU(0.01,inplace=True))
            layers.append(nn.Conv2d(dim[i],dim[i+1],kernel_size=4,stride=2,padding=1,bias=False))
            layers.append(nn.InstanceNorm2d(dim[i+1],affine=True,track_running_stats=True))
            layers.append(nn.LeakyReLU(0.01,inplace=True))

        self.encoder_conv=nn.Sequential(*layers)

        difference=[]
        difference.append(nn.Conv2d(4,32,kernel_size=7,stride=1,padding=0,bias=False))
        # difference.append(nn.LeakyReLU(0.01,inplace=True))
        self.encoder_diff=nn.Sequential(*difference)

    def forward(self,x):
        out=self.encoder_conv(x)
        out_diff=out[:,-4:,:,:]
        out_conv=out[:,:-4,:,:]
        diff=self.encoder_diff(out_diff)
        return out_conv,diff


class decoder(nn.Module):
    def __init__(self):
        super(decoder,self).__init__()

        dim=[36,32,16]
        if(not torch.cuda.is_available()):
            dim=[5,3,3]
        difference=[]
        difference.append(nn.ConvTranspose2d(32,4,kernel_size=7,stride=1,padding=0,bias=True))
        difference.append(nn.InstanceNorm2d(4,affine=True,track_running_stats=True))
        difference.append(nn.LeakyReLU(0.01,inplace=True))
        self.decoder_diff=nn.Sequential(*difference)

        layers=[]
        for i in range(2):
            layers.append(nn.ConvTranspose2d(dim[i],dim[i+1],kernel_size=4,stride=2,padding=1,bias=False))
            layers.append(nn.InstanceNorm2d(dim[i+1],affine=True,track_running_stats=True))
            layers.append(nn.LeakyReLU(0.01,inplace=True))
            # layers.append(nn.ConvTranspose2d(dim[i+1],dim[i+1],kernel_size=3,stride=1,padding=1,bias=False))
            # layers.append(nn.InstanceNorm2d(dim[i+1],affine=True,track_running_stats=True))
            # layers.append(nn.LeakyReLU(0.01,inplace=True))

        layers.append(nn.Conv2d(dim[2],3,kernel_size=5,stride=1,padding=2,bias=False))
        layers.append(nn.Tanh())

        self.decoder_conv=nn.Sequential(*layers)

    def forward(self,conv,diff):
        diff_conv=self.decoder_diff(diff)
        out=torch.cat((conv,diff_conv),1)
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


        layers=[]
        layers.append(nn.Conv2d(32,32,kernel_size=1,stride=1,padding=0))
        layers.append(nn.LeakyReLU(0.01,inplace=True))
        layers.append(nn.Conv2d(32,32,kernel_size=1,stride=1,padding=0))
        layers.append(nn.LeakyReLU(0.01,inplace=True))
        layers.append(nn.Conv2d(32,1,kernel_size=1,stride=1,padding=0))

        self.dis=nn.Sequential(*layers)

    def forward(self,x):
        return self.dis(x)
