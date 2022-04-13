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
        for i in range(data_len - 1):
            for j in range(i + 1, data_len):
                sim_matrix['div'].append(torch.mul(x[i], x[j]).sum(dim=1))

    return sim_matrix


class ABE_loss(nn.Module):
    """
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
        # print(div_mat)

        # contrastive loss
        contrastive_losses = []
        for i in range(m):
            c = 0  # counter
            temp_con_loss = []
            for j in range(n):
                pos_pair = torch.masked_select(metric_mat[i][j], target == target[j])
                neg_pair = torch.masked_select(metric_mat[i][j], target != target[j])

                # remove itself
                pos_pair = torch.masked_select(pos_pair, pos_pair < 1)

                if len(pos_pair) < 1 or len(neg_pair) < 1:
                    c += 1
                    continue
                # old
                pos_loss = torch.mean(F.relu(self.margin_c - pos_pair))
                neg_loss = torch.mean(neg_pair)
                temp_con_loss.append(pos_loss + neg_loss)
                '''
                pos_loss = torch.mean(pos_pair)
                neg_loss = torch.mean(F.relu(neg_pair-self.margin_c))
                temp_con_loss.append(neg_loss-pos_loss)
                '''
            contrastive_losses.append(sum(temp_con_loss) / len(temp_con_loss))
        contrastive_loss = sum(contrastive_losses) / len(contrastive_losses)
        # return contrastive_loss

        assert m > 1
        # divergence loss
        div_losses = []
        for i in range(len(div_mat)):
            temp_div_loss = F.relu(div_mat[i] - self.margin_div).mean()
            div_losses.append(temp_div_loss)
        div_loss = sum(div_losses) / len(div_losses)

        # total_losses
        total_losses = contrastive_loss + self.lambda_div * div_loss
        return total_losses
