import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from reid import create
from reid.embedding import EltwiseSubEmbed
from reid.multi_branch import SiameseNet
import math

def weight_init_normal(m):
    classname=m.__class__.__name__
    if classname.find('Conv')!=-1:
        init.normal(m.weight.data,0.0,0.02)
    elif classname.find('Linear')!=-1:
        init.normal(m.weight.data,0.0,0.02)
    elif classname.find('BatchNorm2d')!=-1:
        init.normal(m.weight.data,1.0,0.02)
        init.constant(m.bias.data,0.0)

def init_weights(net):
    net.apply(weight_init_normal)

class encoder(nn.Module):
    def __init__(self):
        super(encoder,self).__init__()

        self.encoder=create("resnet50", cut_at_pooling=True)

    def forward(self,x):
        out=self.encoder(x)
        same=out[:,-1024:,:,:]
        diff=out[:,:-1024,:,:]
        return same,diff


class decoder(nn.Module):
    def __init__(self,norm_layer=nn.BatchNorm2d,connect_layers=0,dropout=0.2,output_nc=3,use_bias=False):
        super(decoder,self).__init__()
        ngf=64
        self.dropout=dropout
        self.use_bias=use_bias
        input_channel = [[8, 8, 4, 2, 1],
                        [16, 8, 4, 2, 1],
                        [16, 16, 4, 2, 1],
                        [16, 16, 8, 2, 1],
                        [16, 16, 8, 4, 1],
                        [16, 16, 8, 4, 2]]
        de_avg = [nn.ReLU(True),
                    nn.ConvTranspose2d(1024+1024+256, ngf * 8,
                        kernel_size=(8,4), bias=self.use_bias),
                    norm_layer(ngf * 8),
                    nn.Dropout(dropout)]
        self.de_avg = nn.Sequential(*de_avg)
        # N*512*8*4

        self.de_conv5 = self._make_layer_decode(ngf * input_channel[connect_layers][0],ngf * 8)
        # N*512*16*8
        self.de_conv4 = self._make_layer_decode(ngf * input_channel[connect_layers][1],ngf * 4)
        # N*256*32*16
        self.de_conv3 = self._make_layer_decode(ngf * input_channel[connect_layers][2],ngf * 2)
        # N*128*64*32
        self.de_conv2 = self._make_layer_decode(ngf * input_channel[connect_layers][3],ngf)
        # N*64*128*64
        de_conv1 = [nn.ReLU(True),
                    nn.ConvTranspose2d(ngf * input_channel[connect_layers][4],output_nc,
                        kernel_size=4, stride=2,
                        padding=1, bias=self.use_bias),
                    nn.Tanh()]
        self.de_conv1 = nn.Sequential(*de_conv1)
        init_weights(self.de_avg)
        init_weights(self.de_conv5)
        init_weights(self.de_conv4)
        init_weights(self.de_conv3)
        init_weights(self.de_conv2)
        init_weights(self.de_conv1)


    def _make_layer_decode(self, in_nc, out_nc):
        block = [nn.ReLU(True),
                nn.ConvTranspose2d(in_nc, out_nc,
                    kernel_size=4, stride=2,
                    padding=1, bias=self.use_bias),
                self.norm_layer(out_nc),
                nn.Dropout(self.dropout)]
        return nn.Sequential(*block)

    def decode(self, model, fake_feature,  cnlayers):
        if cnlayers>0:
            return model(fake_feature), cnlayers-1
        else:
            return model(fake_feature), cnlayers

    def forward(self,same,diff):
        noise=torch.randn(same.size(0),256)
        out=torch.cat((same,diff,noise),1)

        fake_feature = self.de_avg(out)

        cnlayers = self.connect_layers
        fake_feature_5, cnlayers = self.decode(self.de_conv5, fake_feature, cnlayers)
        fake_feature_4, cnlayers = self.decode(self.de_conv4, fake_feature_5, cnlayers)
        fake_feature_3, cnlayers = self.decode(self.de_conv3, fake_feature_4, cnlayers)
        fake_feature_2, cnlayers = self.decode(self.de_conv2, fake_feature_3, cnlayers)
        fake_feature_1, cnlayers = self.decode(self.de_conv1, fake_feature_2, cnlayers)

        fake_imgs = fake_feature_1
        return fake_imgs

class discriminator_for_image(nn.Module):
    def __init__(self):
        super(discriminator_for_image,self).__init__()
        self.resnet=create("resnet50", cut_at_pooling=True)
        layers=[]
        bn=nn.BatchNorm1d(num_features=2048)
        bn.weight.data.fill_(1)
        bn.bias.data.zero_()
        layers.append(bn)
        layers.append(nn.LeakyReLU(0.01))
        layers.append(nn.Linear(2048,1024))
        bn=nn.BatchNorm1d(num_features=1024)
        bn.weight.data.fill_(1)
        bn.bias.data.zero_()
        layers.append(bn)
        layers.append(nn.LeakyReLU(0.01))
        layers.append(nn.Linear(1024,1))
        self.fclayers=layers
        init_weights(self.fclayers)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        x=self.resnet(x)
        x=self.fclayers(x)
        #x=x.view(x.size(0),-1)
        x=self.sigmoid(x)
        return x


class discriminator_for_difference(nn.Module):
    def __init__(self):
        super(discriminator_for_difference,self).__init__()
        layers=[]
        layers.append(nn.BatchNorm1d(num_features=2048))
        layers.append(nn.LeakyReLU(0.01))
        layers.append(nn.Linear(2048,1024))
        layers.append(nn.BatchNorm1d(num_features=1024))
        layers.append(nn.LeakyReLU(0.01))
        layers.append(nn.Linear(1024,702))
        self.fclayers=layers
        init_weights(self.fclayers)
        self.softmax=F.softmax()

    def forward(self,x):
        x=self.fclayers(x)
        x=x.view(x.size(0),-1)
        return self.softmax(x)


class verification_classifier(nn.Module):
    def __init__(self,num_features=1024,num_classes=1):
        super(verification_classifier,self).__init__()
        self.bn=nn.BatchNorm1d(num_features)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()
        self.classifier = nn.Linear(num_features, num_classes)
        self.classifier.weight.data.normal_(0, 0.001)
        self.classifier.bias.data.zero_()

    def forward(self,x1,x2):
        x=x1-x2
        x=x.pow()
        x=self.bn(x)
        x=x.view(x.size(0),-1)
        x=self.classifier(x)
        return x
