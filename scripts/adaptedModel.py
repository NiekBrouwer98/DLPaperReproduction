import torch
import torch.nn as nn

class Inception(nn.Module):
    def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super().__init__()

        #1x1conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(input_channels, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(inplace=True)
        )

        #1x1conv -> 3x3conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(input_channels, n3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(n3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n3x3_reduce, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(inplace=True)
        )

        #1x1conv -> 5x5conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(input_channels, n5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(n5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5_reduce, n5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True)
        )

        #3x3pooling -> 1x1conv
        #same conv
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(input_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)

class ABE_M(nn.Module):
    def __init__(self, M=8, total_len=512):
        super(ABE_M, self).__init__()
        self.M = M
        self.prelayer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            #nn.Conv2d(64, 64, kernel_size=1, stride=1),
            #nn.BatchNorm2d(64),
            #nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        #although we only use 1 conv layer as prelayer,
        #we still use name a3, b3.......
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        #"""In general, an Inception network is a network consisting of
        #modules of the above type stacked upon each other, with occasional
        #max-pooling layers with stride 2 to halve the resolution of the
        #grid"""
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # attention module
        self.att_a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.att_b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.att_c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.att_d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.att_e4 = Inception(528, 256, 160, 320, 32, 128, 128)
        self.att_conv_branches = []
        for i in range(self.M):
            self.att_conv_branches.append(nn.Sequential(
                nn.Conv2d(832, 480, kernel_size=1,stride=1),
                nn.BatchNorm2d(480),
                nn.ReLU(inplace=True)
            ))
        self.att_conv_branches = nn.ModuleList(self.att_conv_branches)
        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        #input feature size: 7*7*1024
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout2d(p=0.4)
        self.linear = nn.Linear(1024, int(total_len/self.M))

    def forward(self, x):
        output = self.prelayer(x)
        output = self.a3(output)
        output = self.b3(output)

        output = self.maxpool(output)

        att_outputs = []
        for i in range(self.M):
            att_module_output = self.att_a4(output)
            att_module_output = self.att_b4(att_module_output)
            att_module_output = self.att_c4(att_module_output)
            att_module_output = self.att_d4(att_module_output)
            att_module_output = self.att_e4(att_module_output)
            att_mask = self.att_conv_branches[i](att_module_output)

            att_output = torch.mul(att_mask, output)

            att_output = self.a4(att_output)
            att_output = self.b4(att_output)
            att_output = self.c4(att_output)
            att_output = self.d4(att_output)
            att_output = self.e4(att_output)

            att_output = self.maxpool(att_output)

            att_output = self.a5(att_output)
            att_output = self.b5(att_output)

            #"""It was found that a move from fully connected layers to
            #average pooling improved the top-1 accuracy by about 0.6%,
            #however the use of dropout remained essential even after
            #removing the fully connected layers."""
            att_output = self.avgpool(att_output)
            att_output = self.dropout(att_output)
            att_output = att_output.view(att_output.size()[0], -1)
            att_output = self.linear(att_output)
            norm = att_output.norm(dim=1, p=2, keepdim=True)
            att_output = att_output.div(norm.expand_as(att_output))

            att_outputs.append(att_output)

        return att_outputs



# class ABE_M(GoogLeNet.GoogLeNet):
#     def __init__(self, M=8, total_len=512, pretrain=None):
#         super(ABE_M, self).__init__()
#         if pretrain:
#             if os.path.exists(pretrain):
#                 self.load_state_dict(torch.load(pretrain))
#                 print('Loaded pretrained GoogLeNet.')
#             else:
#                 print('Downloading pretrained GoogLeNet.')
#                 state_dict = torch.utils.model_zoo.load_url(
#                     'https://download.pytorch.org/models/googlenet-1378be20.pth')
#                 self.load_state_dict(state_dict)
#         assert 512 % M == 0
#         self.M = M
#         self.out_dim = int(512 / self.M)
#         self.att_depth = 480
#         self.att = nn.ModuleList(
#             [nn.Conv2d(in_channels=832, out_channels=self.att_depth, kernel_size=1, bias=False) for i in
#              range(M)])
#         self.last_fc = nn.Linear(1024, self.out_dim)
#
#
#     def feat_spatial(self, x):
#         # N x 3 x 224 x 224
#         x = self.conv1(x)
#         # N x 64 x 112 x 112
#         x = self.maxpool1(x)
#         # N x 64 x 56 x 56
#         x = self.conv2(x)
#         # N x 64 x 56 x 56
#         x = self.conv3(x)
#         # N x 192 x 56 x 56
#         x = self.maxpool2(x)
#
#         # N x 192 x 28 x 28
#         x = self.inception3a(x)
#         # N x 256 x 28 x 28
#         x = self.inception3b(x)
#         # N x 480 x 28 x 28
#         x = self.maxpool3(x)
#
#         return x
#
#     def feat_global(self, x):
#         # N x 480 x 14 x 14
#         x = self.inception4a(x)
#         # N x 512 x 14 x 14
#         x = self.inception4b(x)
#         # N x 512 x 14 x 14
#         x = self.inception4c(x)
#         # N x 512 x 14 x 14
#         x = self.inception4d(x)
#         # N x 528 x 14 x 14
#         if self.training and self.aux_logits:
#             aux2 = self.aux2(x)
#
#         x = self.inception4e(x)
#         # N x 832 x 14 x 14
#         x = self.maxpool4(x)
#         # N x 832 x 7 x 7
#         x = self.inception5a(x)
#         # N x 832 x 7 x 7
#         x = self.inception5b(x)
#         # N x 1024 x 7 x 7
#
#         x = self.avgpool(x)
#         # N x 1024 x 1 x 1
#         x = torch.flatten(x, 1)
#         # N x 1024
#         # x = self.dropout(x)
#         # N x 1024
#         x = self.last_fc(x)
#         # N x (512/M)
#         return x
#
#     def att_prep(self, x):
#         # N x 480 x 14 x 14
#         a4 = self.inception4a(x)
#         # N x 512 x 14 x 14
#         b4 = self.inception4b(a4)
#         # N x 512 x 14 x 14
#         c4 = self.inception4c(b4)
#         # N x 512 x 14 x 14
#         d4 = self.inception4d(c4)
#         # N x 528 x 14 x 14
#         e4 = self.inception4e(d4)
#         # N x 832 x 14 x 14
#         return e4
#
#     def forward(self, x, sampling=True):
#         # N x 3 x 224 x 224
#         sp = self.feat_spatial(x)
#         att_input = self.att_prep(sp)
#         atts = torch.cat([self.att[i](att_input).unsqueeze(1) for i in range(self.M)],
#                          dim=1)  # (N, att_heads, depth, H, W)
#         # Normalize attention map
#         N, _, D, H, W = atts.size()
#         atts = atts.view(-1, H * W)  # (N*att_heads*depth, H*W)
#         att_max, _ = atts.max(dim=1, keepdim=True)  # (N*att_heads*depth, 1)
#         att_min, _ = atts.min(dim=1, keepdim=True)  # (N*att_heads*depth, 1)
#         atts = (atts - att_min) / (att_max - att_min)  # (N*depth, H*W)
#         atts = atts.view(N, -1, D, H, W)
#
#         embedding = torch.cat([self.feat_global(atts[:, i, ...] * sp).unsqueeze(1) for i in range(self.M)], 1)
#         embedding = torch.flatten(embedding, 1)
#
#         # #return embedding
#         # if sampling:
#         #     sampled = DistanceWeightedSampling(batch_k=4)
#         #     return sampled(embedding)
#
#         return embedding

# def l2_norm(x):
#     if len(x.shape):
#         x = x.reshape((x.shape[0],-1))
#     return F.normalize(x, p=2, dim=1)
#
#
# def get_distance(x):
#     _x = x.detach()
#     sim = torch.matmul(_x, _x.t())
#     sim = torch.clamp(sim, max=1.0)
#     #print('\n\n', np.count_nonzero(sim.cpu().numpy() > 0.9) / (sim.shape[0] * sim.shape[1]), sim.shape)
#     dist = 2 - 2*sim
#     dist += torch.eye(dist.shape[0]).to(dist.device)   # maybe dist += torch.eye(dist.shape[0]).to(dist.device)*1e-8
#     dist = dist.sqrt()
#     return dist



# class   DistanceWeightedSampling(nn.Module):
#
#     # def __init__(self):
#     #     pass
#     #
#     # # todo: generate right sampling method
#
#     '''
#     parameters
#     ----------
#     batch_k: int
#         number of images per class
#     Inputs:
#         data: input tensor with shapeee (batch_size, edbed_dim)
#             Here we assume the consecutive batch_k examples are of the same class.
#             For example, if batch_k = 5, the first 5 examples belong to the same class,
#             6th-10th examples belong to another class, etc.
#     Outputs:
#         a_indices: indicess of anchors
#         x[a_indices]
#         x[p_indices]
#         x[n_indices]
#         xxx
#     '''
#
#     def __init__(self, batch_k, cutoff=0.5, nonzero_loss_cutoff=1.4, normalize =False,  **kwargs):
#         super(DistanceWeightedSampling,self).__init__()
#         self.batch_k = batch_k
#         self.cutoff = cutoff
#         self.nonzero_loss_cutoff = nonzero_loss_cutoff
#         self.normalize = normalize
#
#     def forward(self, x):
#         k = self.batch_k
#         n, d = x.shape
#         distance = get_distance(x)
#         distance = distance.clamp(min=self.cutoff)
#         log_weights = ((2.0 - float(d)) * distance.log() - (float(d-3)/2)*torch.log(torch.clamp(1.0 - 0.25*(distance*distance), min=1e-8)))
#
#         if self.normalize:
#             log_weights = (log_weights - log_weights.min()) / (log_weights.max() - log_weights.min() + 1e-8)
#
#         weights = torch.exp(log_weights - torch.max(log_weights))
#
#         if x.device != weights.device:
#             weights = weights.to(x.device)
#
#         mask = torch.ones_like(weights)
#         for i in range(0,n,k):
#             mask[i:i+k, i:i+k] = 0
#
#         mask_uniform_probs = mask.double() *(1.0/(n-k))
#
#         weights = weights*mask*((distance < self.nonzero_loss_cutoff).float()) + 1e-8
#         weights_sum = torch.sum(weights, dim=1, keepdim=True)
#         weights = weights / weights_sum
#
#         a_indices = []
#         p_indices = []
#         n_indices = []
#
#         np_weights = weights.cpu().numpy()
#         for i in range(n):
#             block_idx = i // k
#
#             if weights_sum[i] != 0:
#                 n_indices +=  np.random.choice(n, k-1, p=np_weights[i]).tolist()
#             else:
#                 n_indices +=  np.random.choice(n, k-1, p=mask_uniform_probs[i]).tolist()
#             for j in range(block_idx * k, (block_idx + 1)*k):
#                 if j != i:
#                     a_indices.append(i)
#                     p_indices.append(j)
#
#         return  a_indices, x[a_indices], x[p_indices], x[n_indices], x
