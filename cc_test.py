# %matplotlib widget
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms


import cc

# download dataset
transform = transforms.Compose(
    [transforms.Resize(32),
     transforms.ToTensor(),
     transforms.Normalize(0.5, 0.5)])
dataset = datasets.MNIST(root='./torch-dataset', train=True,
                         download=True, transform=transform)

print('Loading data...')
# load dataset into pytorch
data_loader = torch.utils.data.DataLoader(dataset, batch_size=600000)
data,labels = next(iter(data_loader))
data = data.cuda()

# select single class of dataset
Z = data[labels==2]
Z = Z.reshape((5958,32**2))
Z = Z.T

print('Starting CC!')
# main training command
cc.cc(Z, k=24)


