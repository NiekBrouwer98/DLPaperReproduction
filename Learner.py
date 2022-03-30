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


    def a4_to_e4(self, x):
        # N x 480 x 14 x 14
        a4 = self.inception4a(x)
        # N x 512 x 14 x 14
        b4 = self.inception4b(a4)
        # N x 512 x 14 x 14
        c4 = self.inception4c(b4)
        # N x 512 x 14 x 14
        d4 = self.inception4d(c4)
        # N x 528 x 14 x 14
        e4 = self.inception4e(d4)
        # N x 832 x 14 x 14      
        return e4

    def forward(self, x, ret_att=False, sampling=True):
        # N x 3 x 224 x 224
        sp = self.feat_spatial(x)
        # output of pool3
        att_input = self.a4_to_e4(sp)
        atts = [self.att[i](att_input) for i in range(self.att_heads)] # (N, att_heads, depth, H, W)
        # Normalize attention map
        for i in range(len(atts)):
            N, D, H, W = atts[i].size()
            att = atts[i].view(-1, H*W)
            att_max, _ = att.max(dim=1, keepdim=True)
            att_min, _ = att.min(dim=1, keepdim=True)
            atts[i] = ((att - att_min) / (att_max - att_min)).view(N, D, H, W)
        
        embedding = torch.cat([self.feat_global(atts[i]*sp).unsqueeze(1) for i in range(self.att_heads)], 1)
        embedding = torch.flatten(embedding, 1)
        if sampling:
            return self.sampled(embedding) if not ret_att else (self.sampled(embedding), atts)
        else:
            return (embedding, atts) if ret_att else embedding


    def l2_norm(x):
        if len(x.shape):
            x = x.reshape((x.shape[0],-1))
        return F.normalize(x, p=2, dim=1)


    def get_distance(x):
        _x = x.detach()
        sim = torch.matmul(_x, _x.t())
        sim = torch.clamp(sim, max=1.0)
        dist = 2 - 2*sim
        dist += torch.eye(dist.shape[0]).to(dist.device)   # maybe dist += torch.eye(dist.shape[0]).to(dist.device)*1e-8
        dist = dist.sqrt()
        return dist


