import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

def get_sim_matrix(x):

    data_len = len(x)
    sim_matrix = defaultdict(list)
    for i in range(data_len):
        sim_matrix['metric'].append(torch.matmul(x[i], x[i].t()))
    if data_len > 1:
        for i in range(data_len-1):
            for j in range(i+1, data_len):
                sim_matrix['div'].append(torch.mul(x[i],x[j]).sum(dim=1))

    return sim_matrix


class ABE_loss(nn.Module):
    """
    this loss function is used in ABE-M paper
    the loss contains two parts:
        - constrastive loss
        - divergence loss
    """
    def __init__(self, lambda_div=0.05, margin_c=0.5, margin_div=0.2):
        super(ABE_loss, self).__init__()
        self.lambda_div = lambda_div
        self.margin_c = margin_c
        self.margin_div = margin_div

    def forward(self, output, target):
        m = len(output)
        n = output[0].size(0)

        sim_mat = get_sim_matrix(output)
        metric_mat = sim_mat['metric']
        div_mat = sim_mat['div']
        #print(div_mat)

        #contrastive loss
        contrastive_losses = []
        for i in range(m):
            c = 0   # counter
            temp_con_loss = []
            for j in range(n):
                pos_pair = torch.masked_select(metric_mat[i][j], target==target[j])
                neg_pair = torch.masked_select(metric_mat[i][j], target!=target[j])

                # remove itself
                pos_pair = torch.masked_select(pos_pair, pos_pair<1)

                if len(pos_pair)<1 or len(neg_pair)<1:
                    c += 1
                    continue
                # old
                pos_loss = torch.mean(F.relu(self.margin_c-pos_pair))    ##
                neg_loss = torch.mean(neg_pair)                  ##
                temp_con_loss.append(pos_loss+neg_loss)
                '''
                pos_loss = torch.mean(pos_pair)
                neg_loss = torch.mean(F.relu(neg_pair-self.margin_c))
                temp_con_loss.append(neg_loss-pos_loss)
                '''
            contrastive_losses.append(sum(temp_con_loss)/len(temp_con_loss))
        contrastive_loss = sum(contrastive_losses)/len(contrastive_losses)
        #return contrastive_loss

        assert m > 1
        #divergence loss
        div_losses = []
        for i in range(len(div_mat)):
            temp_div_loss = F.relu(div_mat[i] - self.margin_div).mean()
            div_losses.append(temp_div_loss)
        div_loss = sum(div_losses)/len(div_losses)

        #total_losses
        total_losses = contrastive_loss+self.lambda_div*div_loss
        return total_losses

# def L_metric(feat1, feat2, same_class=True):
#     '''
#         feat1 same size as feat2
#         feat size: (batch_size, atts, feat_size)
#     '''
#     d = torch.sum((feat1 - feat2).pow(2).view((-1, feat1.size(-1))), 1)
#     if same_class:
#         return d.sum() / d.size(0)
#     else:
#         return torch.clamp(1-d, min=0).sum() / d.size(0)
#
# def L_divergence(feats):
#     n = feats.shape[0]
#     loss = 0
#     cnt = 0
#     for i in range(n):
#         for j in range(i+1, n):
#             loss += torch.clamp(1-torch.sum((feats[i, :] - feats[j, :]).pow(2)), min=0)
#             cnt += 1
#     return loss / cnt
#
# def loss_func(tensor, batch_k):
#         batch_size = tensor.size(0)
#         assert batch_size % batch_k == 0
#         assert batch_k > 1
#         loss_homo, loss_heter, loss_div = 0, 0, 0
#         for i in range(batch_size):
#                 loss_div += L_divergence(tensor[i, ...])
#
#         cnt_homo, cnt_heter = 0, 0
#         for group_index in range(batch_size // batch_k):
#                 for i in range(batch_k):
#                         anchor = tensor[i+group_index*batch_k: 1+i+group_index*batch_k, ...]
#                         for j in range(i+1, batch_k):
#                                 index = j+group_index*batch_k
#                                 loss_homo += L_metric(anchor, tensor[index: 1+index, ...])
#                                 cnt_homo += 1
#                         for j in range((group_index+1)*batch_k, batch_size):
#                                 loss_heter += L_metric(anchor, tensor[j:j+1, ...], same_class=False)
#                                 cnt_heter += 1
#         return loss_div/batch_size, loss_homo/cnt_homo, loss_heter/cnt_heter
#
# def criterion(anchors, positives, negatives):
#         loss_homo = L_metric(anchors, positives)
#         loss_heter = L_metric(anchors, negatives, False)
#         loss_div = 0
#         for i in range(anchors.shape[0]):
#                 loss_div += (L_divergence(anchors[i, ...]) + L_divergence(positives[i, ...]) + L_divergence(negatives[i, ...])) / 3
#         return loss_div / anchors.shape[0], loss_homo, loss_heter
