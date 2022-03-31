import torch
import torch.nn.functional as F

def L_metric(feat1, feat2, same_class=True):
    '''
        feat1 same size as feat2
        feat size: (batch_size, atts, feat_size)
    '''
    d = torch.sum((feat1 - feat2).pow(2).view((-1, feat1.size(-1))), 1)
    if same_class:
        return d.sum() / d.size(0)
    else:
        return torch.clamp(1-d, min=0).sum() / d.size(0)

def L_divergence(feats):
    n = feats.shape[0]
    loss = 0
    cnt = 0
    for i in range(n):
        for j in range(i+1, n):
            loss += torch.clamp(1-torch.sum((feats[i, :] - feats[j, :]).pow(2)), min=0)
            cnt += 1
    return loss / cnt

def loss_func(tensor, batch_k):
        batch_size = tensor.size(0)
        assert batch_size % batch_k == 0
        assert batch_k > 1
        loss_homo, loss_heter, loss_div = 0, 0, 0
        for i in range(batch_size):
                loss_div += L_divergence(tensor[i, ...])

        cnt_homo, cnt_heter = 0, 0
        for group_index in range(batch_size // batch_k):
                for i in range(batch_k):
                        anchor = tensor[i+group_index*batch_k: 1+i+group_index*batch_k, ...]
                        for j in range(i+1, batch_k):
                                index = j+group_index*batch_k
                                loss_homo += L_metric(anchor, tensor[index: 1+index, ...])
                                cnt_homo += 1
                        for j in range((group_index+1)*batch_k, batch_size):
                                loss_heter += L_metric(anchor, tensor[j:j+1, ...], same_class=False)
                                cnt_heter += 1
        return loss_div/batch_size, loss_homo/cnt_homo, loss_heter/cnt_heter   

def criterion(anchors, positives, negatives):
        loss_homo = L_metric(anchors, positives)
        loss_heter = L_metric(anchors, negatives, False)
        loss_div = 0
        for i in range(anchors.shape[0]):
                loss_div += (L_divergence(anchors[i, ...]) + L_divergence(positives[i, ...]) + L_divergence(negatives[i, ...])) / 3
        return loss_div / anchors.shape[0], loss_homo, loss_heter
