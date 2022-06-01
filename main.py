from Network import LeNet5
from init import init_dataloader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
import numpy as np
import os
from PIL import Image
from preprocess import reinforcement, vampix
import statistics


def train(epoch, network, optimizer, train_loader):
    network.train()  # 使神经网络处于训练模式
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # 手动将梯度设置为0
        output = network(data)  # 前向传递网络的输出
        loss = F.nll_loss(output, target)  # 也可以用CrossEntropyLoss一步到位
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            torch.save(network.state_dict(), 'pth/model.pth')
            torch.save(optimizer.state_dict(), 'pth/optimizer.pth')


def test(model, test_loader):
    model.eval()
    test_loss = 0

    statistics.digital_num(test_loader)
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
    test_loss /= len(test_loader.dataset)
    print('\nTest set\nAverage loss: {:.4f}\n'.format(test_loss))

    pred = pred.squeeze(1)
    statistics.macro_Avg(target, pred)
    statistics.micro_Avg(target, pred)




def predict_image(path, model):
    img = Image.open(path)
    # img.show()
    show = ToPILImage()  # 可以把Tensor转成Image，方便可视化
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.ToTensor(),
                                    transforms.ColorJitter(contrast=100),])
    trans1 = nn.ReplicationPad2d(2)
    trans2 = transforms.Resize((28, 28))

    tensor = reinforcement(transform(img))
    tensor = trans1(trans2(tensor))
    tensor = vampix(reinforcement(tensor))
    show(tensor).show()

    temp = tensor.data.cpu().numpy()
    m, s = np.mean(temp), np.std(temp)
    tensor = (tensor - m) / s

    tensor = tensor.unsqueeze(dim=0)
    pred = model(tensor).data.max(1, keepdim=True)[1]
    print("The result of prediction is：{}".format(pred[0][0]))




if __name__ == '__main__':
    n_epochs = 20
    network = LeNet5()
    train_loader, test_loader = init_dataloader()
    if os.path.exists('pth/model.pth') and os.path.exists('pth/optimizer.pth'):
        network.load_state_dict(torch.load('pth/model.pth'))
        optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.5)
        optimizer.load_state_dict(torch.load('pth/optimizer.pth'))
        test(model=network, test_loader=test_loader)
        # train(1, network, optimizer, train_loader=train_loader)
        predict_image('example/7_2.png', network)
    else:
        optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.5)
        for i in range(1, 10):
            train(i, network, optimizer, train_loader=train_loader)
            test(model=network, test_loader=test_loader)

