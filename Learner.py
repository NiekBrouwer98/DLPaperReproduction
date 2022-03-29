import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
import GoogLeNet
import os
import GoogleLeNet


class IndividualLearner(GoogleLeNet.GoogLeNet):
    def __init__(self, attention_heads, pretrain, batch_size, normalize):
        super(IndividualLearner, self).__init__()
        self.load_state_dict(torch.load(pretrain))
        self.attention_heads = attention_heads
        self.embed_size = 512
        self.kernels = 480
        self.out_dim = int(self.embed_size/self.attention_heads)
        self.attention = nn.ModuleList([nn.conv2d(in_channels = 832, out_channels=self.kernels, kernel_size =1, bias= False)] for i in range(attention_heads)) #why use 832 as in_channel size?
        self.forward_call = nn.Linear(1024, self.out_dim)

    #Spatial Features from h_l to h_{i+1}
    def feat_spatial(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        return x

    #Global Features from h_{i} to h_1
    def feat_global(self, x):
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        #x = self.dropout(x)
        x = self.forward_call(x)
        # N x 1000 (num_classes)
        x = F.normalize(x)
        return x

