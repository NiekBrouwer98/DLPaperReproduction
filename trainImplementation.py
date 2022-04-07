import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from DLPaperReproduction.modelImplementation import ABE_M
from criterion import ABE_loss
from sampler_excelfile import SourceSampler
from sampler_excelfile import MetricData


def train(train_loader, model, loss_fn, optimizer, device):
    """
    Method that trains the model

    args:
        train_loader = dataset
        model = network
        loss_fn = loss function, can be find in class criterion
        optimizer
        device = cpu/gpu
    """

    model.train()
    losses = []
    total_loss = 0
    for i, batch in enumerate(train_loader):
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        output = model(x)
        target = y

        loss_outputs = loss_fn(output, target)

        losses.append(loss_outputs.item())
        total_loss += loss_outputs.item()
        if loss_outputs.requires_grad is True:
            loss_outputs.backward()
            optimizer.step()

        if i % 20 == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(i * target.size(0), len(train_loader.dataset),
                                                                      100. * i / len(train_loader), np.mean(losses))
            print(message)
            losses = []

    print('total loss {:.6f}'.format(total_loss / (i + 1)))


def test(test_loader, model, loss_fn, device, best):
    """
    Method that evaluates the trained model

    args:
        test_loader = dataset
        model = network
        loss_fn = loss function
        device
        best = current best
    """

    with torch.no_grad():
        model.eval()
        val_loss = 0
        for batch_idx, (data, target, _) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device)

            output = model(data)

            loss_outputs = loss_fn(output, target)
            # print(loss_outputs)
            val_loss += loss_outputs.item()

    print('val loss {:.6f}'.format(val_loss / (batch_idx + 1)))
    if best > val_loss / (batch_idx + 1):
        best = val_loss / (batch_idx + 1)
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), './model/score{:.4f}.pth'.format(best))
        else:
            torch.save(model.state_dict(), './model/score{:.4f}.pth'.format(best))
    return best


def main():
    """"
    Main method looping over both training and testing
    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    epochs = 100
    lr = 0.001
    model = ABE_M()
    if torch.cuda.device_count() > 1:
        print("Let's use {} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    loss_fn = ABE_loss()

    datatrain = MetricData(data_root='data/CUB_100_train',
                           anno_file=r'data/annos_traindataset.xlsx',
                           idx_file='data/idx_trainset.pkl',
                           return_fn=True)

    samplertrain = SourceSampler(datatrain)
    print('Batch sampler len:', len(samplertrain))
    traindata_loader = torch.utils.data.DataLoader(datatrain, batch_sampler=samplertrain)

    datatest = MetricData(data_root='data/CUB_100_test',
                          anno_file=r'data/annos_testdataset.xlsx',
                          idx_file='data/idx_testset.pkl',
                          return_fn=True)

    samplertest = SourceSampler(datatest)
    print('Batch sampler len:', len(samplertest))
    testdata_loader = torch.utils.data.DataLoader(datatest, batch_sampler=samplertest)

    best = 10000000
    for epoch in range(epochs):
        print('epoch {}/{}'.format(epoch, epochs))
        train(traindata_loader, model, loss_fn, optimizer, device)
        best = test(testdata_loader, model, loss_fn, device, best)


if __name__ == '__main__':
    main()
