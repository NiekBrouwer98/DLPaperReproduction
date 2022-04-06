import shutil
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from adaptedModel import ABE_M
import criterion
from sampler_excelfile import SourceSampler
from sampler_excelfile import MetricData

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
        torch.save({'state_dict': model.cpu().state_dict(), 'epoch': epoch+1, 'loss': best}, \
                   os.path.join('../ckpt', '%d_ckpt.pth' % epoch))
        shutil.copy(os.path.join('../ckpt', '%d_ckpt.pth' % epoch), os.path.join('../ckpt', 'best_performance.pth'))
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
    epochs = 1
    batch_size = 32
    model = ABE_M()
    model.to(device)
    lr = 1e-4
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    datatrain = MetricData(data_root='data/CUB_100_train', \
                           anno_file=r'../data/annos_traindataset.xlsx', \
                           idx_file='../data/idx_trainset.pkl', \
                           return_fn=True)
    
    samplertrain = SourceSampler(datatrain)
    print('Batch sampler len:', len(samplertrain))
    traindata_loader = torch.utils.data.DataLoader(datatrain, batch_sampler=samplertrain)
    
    datatest = MetricData(data_root='data/CUB_100_test', \
                          anno_file=r'../data/annos_testdataset.xlsx', \
                          idx_file='../data/idx_testset.pkl', \
                          return_fn=True)
    
    samplertest = SourceSampler(datatest)
    print('Batch sampler len:', len(samplertest))
    testdata_loader = torch.utils.data.DataLoader(datatest, batch_sampler=samplertest)

    best = 10000000
    for epoch in range(epochs):
        print('epoch {}/{}'.format(epoch, epochs))
        train_epoch(traindata_loader, testdata_loader, model, best, optimizer, device)


