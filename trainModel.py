import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from DLPaperReproduction.model import ABE_M
from criterion import ABE_loss
from sampler_excelfile import SourceSampler
from sampler_excelfile import MetricData
from tensorboardX import SummaryWriter


def train(train_loader, model, loss_fn, optimizer, device):
    """
    Method that trains the model

    args:
        train_loader: dataset
        model: network
        loss_fn: loss function, can be find in class criterion
        optimizer
        device: cpu/gpu
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

        if i % 100 == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(i * target.size(0), len(train_loader.dataset),
                                                                      100. * i / len(train_loader), np.mean(losses))
            print(message)

            losses = []

    print('total loss {:.6f}'.format(total_loss / (i + 1)))

    return total_loss / (i + 1)


def test(test_loader, model, loss_fn, device, best):
    """
    Method that evaluates the trained model

    args:
        test_loader: dataset
        model: network
        loss_fn: loss function
        device
        best: current best
    """

    with torch.no_grad():
        model.eval()
        val_loss = 0
        val_losses = []
        for i, batch in enumerate(test_loader):
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            output = model(x)

            loss_outputs = loss_fn(output, y)
            val_losses.append(loss_outputs.item())
            val_loss += loss_outputs.item()

            # if i % 100 == 0:
            #     writer.add_scalars(main_tag='Validate', tag_scalar_dict={'loss': np.mean(val_losses)},global_step=i)
            #
            #     val_losses = []

    print('val loss {:.6f}'.format(val_loss / (i + 1)))
    if best > val_loss / (i + 1):
        best = val_loss / (i + 1)
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), './model/score_{:.4f}.pth'.format(best))
        else:
            torch.save(model.state_dict(), './model/score_{:.4f}.pth'.format(best))

    val_loss = val_loss / (i+1)
    return best, val_loss


def main():
    """"
    Main method looping over both training and testing
    """
    writer = SummaryWriter()

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

    datatrain = MetricData(data_root='CUB_100_train',
                           anno_file=r'./data/annos_traindataset.xlsx',
                           idx_file='./data/idx_trainset.pkl',
                           return_fn=True)

    samplertrain = SourceSampler(datatrain)
    print('Training batch sampler len:', len(samplertrain))
    traindata_loader = torch.utils.data.DataLoader(datatrain, batch_sampler=samplertrain)

    datatest = MetricData(data_root='CUB_100_test',
                          anno_file=r'./data/annos_testdataset.xlsx',
                          idx_file='./data/idx_testset.pkl',
                          return_fn=True)

    samplertest = SourceSampler(datatest)
    print('Test batch sampler len:', len(samplertest))
    testdata_loader = torch.utils.data.DataLoader(datatest, batch_sampler=samplertest)

    best = 10000000
    for epoch in range(epochs):
        print('epoch {}/{}'.format(epoch, epochs))
        train_loss = train(traindata_loader, model, loss_fn, optimizer, device)
        writer.add_scalars(main_tag='Train', tag_scalar_dict={'loss': train_loss}, global_step=epoch)

        best, val_loss = test(testdata_loader, model, loss_fn, device, best)
        writer.add_scalars(main_tag='Validate', tag_scalar_dict={'loss': val_loss}, global_step=epoch)

    print('Finished Training')
    writer.flush()
    writer.close()


if __name__ == '__main__':
    # main()
    from tensorboard import program

    tracking_address = "./runs" # the path of your log file.

    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
