import sys
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from adaptedModel import ABE_M
import criterion
import dataset
from learningrate import find_lr
from sampler import BalancedBatchSampler
import dill
import pickle


def train_epoch(train_loader, eval_loader, model, best, optimizer, device):
    model.train()
    loss_div, loss_homo, loss_heter = 0, 0, 0
    for i, batch in enumerate(train_loader):
        x, y = batch
        x = x.to(device)
        out = model(x, sampling=True)
        a_indices, anchors, positives, negatives, _ = out
        anchors = anchors.view(anchors.size(0), 4, -1)
        positives = positives.view(positives.size(0), 4, -1)
        negatives = negatives.view(negatives.size(0), 4, -1)

        optimizer.zero_grad()
        l_div, l_homo, l_heter = criterion.criterion(anchors, positives, negatives)
        l = l_div + l_homo + l_heter
        l.backward()
        optimizer.step()

        loss_homo += l_homo.item()
        loss_heter += l_heter.item()
        loss_div += l_div.item()

    loss_homo /= (i+1)
    loss_heter /= (i+1)
    loss_div /= (i+1)
    print('Epoch %d batches %d\tdiv:%.4f\thomo:%.4f\theter:%.4f'%(epoch, i+1, loss_div, loss_homo, loss_heter))
    if (loss_homo+loss_heter+loss_div) < best:
        best = loss_homo + loss_heter + loss_div
        torch.save({'state_dict': model.cpu().state_dict(), 'epoch': epoch+1, 'loss': best},
                   './model/{}_{:.4f}.pth'.format("ABE-M", best))
        print('saved model')
        model.to(device)

    model.eval()
    loss_div, loss_homo, loss_heter = 0, 0, 0
    for i, batch in enumerate(eval_loader):
        x, y = batch
        x = x.to(device)
        with torch.no_grad():
            embeddings = model(x, sampling=False)
        embeddings = embeddings.view(embeddings.size(0), 4, -1)
        l_div, l_homo, l_heter = criterion.loss_func(embeddings, 4)
        loss_homo += l_homo.item()
        loss_heter += l_heter.item()
        loss_div += l_div.item()

    print('\tTest phase %d samples\tloss div: %.4f (%.3f)\tloss homo: %.4f (%.3f)\tloss heter: %.4f (%.3f)'%\
        (i, l_div.item(), loss_div/(i+1), l_homo.item(), loss_homo/(i+1), l_heter.item(), loss_heter/(i+1)))


if __name__ == '__main__':
    device = torch.device('cpu')
    epochs = 100
    batch_size = 32
    model = ABE_M()
    model.to(device)
    lr = 1e-4
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    traindata = dataset.main_train()
    train_dataloader = DataLoader(traindata, batch_sampler=BalancedBatchSampler(traindata,batch_size=batch_size, batch_k=4, length=2000), num_workers=4)

    testdata = dataset.main_test()
    test_dataloader = DataLoader(testdata, batch_sampler=BalancedBatchSampler(testdata,batch_size=batch_size, batch_k=4, length=5000//2))

    best = 10000000
    for epoch in range(epochs):
        print('epoch {}/{}'.format(epoch, epochs))
        train_epoch(train_dataloader, test_dataloader, model, best, optimizer, device)


