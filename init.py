import torch
import torch.utils.data as tData
from torchvision import datasets, transforms
import numpy as np


def init_dataloader():
    torch.manual_seed(1)
    mnist = datasets.MNIST('./data/', train=True, download=True,
                           transform=transforms.Compose([transforms.ToTensor(), ]))
    data = [d[0].data.cpu().numpy() for d in mnist]
    mean, std = np.mean(data), np.std(data)
    print("Train set\nmean: {}\nstandard deviation: {}".format(mean, std))  # 使训练数据更快更容易收敛

    train_loader = tData.DataLoader(
        datasets.MNIST('./data/', train=True, download=False,
                       transform=transforms.Compose([  # 将多个图片变换方式结合在一起
                           transforms.ToTensor(),
                           transforms.RandomCrop(size=(32, 32), padding=2),
                           transforms.Normalize((mean,), (std,))])
                       ),
        batch_size=50, shuffle=True)

    test_loader = tData.DataLoader(
        datasets.MNIST('./data/', train=False, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.RandomCrop(size=(32, 32), padding=2),
                           transforms.Normalize((mean,), (std,))])
                       ),
        batch_size=1000, shuffle=True)
    return train_loader, test_loader