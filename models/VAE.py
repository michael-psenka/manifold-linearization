#from https://github.com/lyeoni/pytorch-mnist-VAE/blob/master/pytorch-mnist-VAE.ipynb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image

bs = 100
# MNIST Dataset
train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, h_dim3, z_dim):
        super(VAE, self).__init__()
    
        self.encode = nn.Sequential(# 28 -> 14
            nn.Conv2d(x_dim, h_dim1, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(h_dim1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(h_dim1, h_dim2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(h_dim2),
            nn.LeakyReLU(0.1),
            nn.Linear(h_dim2, h_dim3),
            nn.BatchNorm1d(h_dim3),
            nn.LeakyReLU(0.1),
        )
        self.fc31 = nn.Sequential(
            nn.Linear(h_dim3, z_dim),
            nn.Sigmoid(),
        )
        self.fc32 = nn.Sequential(
            nn.Linear(h_dim3, z_dim),
            nn.Sigmoid(),
        )

        self.decode =  nn.Sequential(
            nn.Linear(z_dim, h_dim3),
            nn.BatchNorm1d(h_dim3),
            nn.ReLU(),
            nn.Linear(h_dim3, h_dim2),
            nn.BatchNorm1d(h_dim2),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim2, h_dim1, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(h_dim1),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim1, x_dim, 4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )



    def encoder(self, x):
        h = self.encode(x)
        return self.fc31(h), self.fc32(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        return self.decode(z)
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var


# build model
vae = VAE(x_dim=784, h_dim1= 1024, h_dim2=7*7*128, h_dim3 = 64, z_dim=2)
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
vae.to(device)

optimizer = optim.Adam(vae.parameters())
# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

vae.train()
train_loss = 0
for batch_idx, (data, _) in enumerate(train_loader):
    data = data.to(device)
    optimizer.zero_grad()
    
    recon_batch, mu, log_var = vae(data)
    loss = loss_function(recon_batch, data, mu, log_var)
    
    loss.backward()
    train_loss += loss.item()
    optimizer.step()
    
    if batch_idx % 100 == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item() / len(data)))
print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
torch.save('vae.dat', vae)