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
# ############ MNIST
# load dataset into pytorch
# data_loader = torch.utils.data.DataLoader(dataset, batch_size=600000)
# data,labels = next(iter(data_loader))
# data = data.cuda()

# # select single class of dataset
# Z = data[labels==2]
# Z = Z.reshape((5958,32**2))
# Z = Z.T

# ############ dummy data
# Z = torch.randn(10,1000)
d = 2
n = 25
Z =torch.zeros((d,n))
for i in range(n):
	x = (1.5 - 1.4/2 +1.4*(i+1)/n)*torch.pi
	Z[0,i] = torch.cos(torch.tensor(x))
	# Z[1,i] = scale*np.sin(x)
	Z[1,i] = torch.sin(torch.tensor(x))


# center and scale
Z_mean_orig = Z.mean(axis=1, keepdim=True)
Z = Z - Z_mean_orig
Z_norm_orig = np.linalg.norm(Z, 'fro')
Z = Z * n / Z_norm_orig
print('Starting CC!')
# main training command
f = cc.cc(Z, d_desired=1)

Z_new = f(Z).detach().numpy()

plt.plot(Z[0,:], Z[1,:], '.')
plt.show()
plt.plot(Z_new[0,:], Z_new[1,:], '.', c='r')
plt.show()


