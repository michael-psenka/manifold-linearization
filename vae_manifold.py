from torch.utils import data
import train_vae
import torch
import torch.nn.functional as F
import os
import numpy as np
from torchvision import datasets, transforms
import flatnet

from flatnet.modules import flatnet_nn
from mnist_test import dim_svd, load_weights,in_F,pre_f
import warnings
warnings.filterwarnings('ignore')




def loss_fn(x, x_hat, mu, log_std):
    recon_loss = F.mse_loss(x, x_hat, reduction='none').view(x.shape[0], -1).sum(1).mean()
    kl_loss = -log_std - 0.5 + (torch.exp(2 * log_std) + mu ** 2) * 0.5
    kl_loss = kl_loss.sum(1).mean()
    loss = recon_loss + kl_loss
    print(f'loss: {loss.item()}, recon_loss: {recon_loss.item()}, kl_loss: {kl_loss.item()}')

transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.CIFAR10('data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10('data', train=False, download=True, transform=transform)

# convert CIFAR10 objects to numpy arrays
train_data = np.array(train_data.data)
test_data = np.array(test_data.data)

train_data = (np.transpose(train_data, (0, 3, 1, 2)) / 255.).astype('float32')
test_data = (np.transpose(test_data, (0, 3, 1, 2)) / 255.).astype('float32')

model=train_vae.ConvVAE((3, 32, 32), 32).cuda()
# load weights
model.load_state_dict(torch.load(os.path.join('vae_weights', 'vae_model.pth')))

train_loader = data.DataLoader(train_data, batch_size=2000, shuffle=True,generator=torch.Generator(device='cuda'))
test_loader = data.DataLoader(test_data, batch_size=1500)

train_data = next(iter(train_loader))
test_data = next(iter(test_loader))

train_data = train_data.cuda()

model.eval()
model.requires_grad_(False)

mu, log_std = model.encoder(train_data)
z= torch.randn_like(mu) * log_std.exp() + mu
X_hat = model.decoder(z)
# loss_fn(train_data, X_hat, mu, log_std)
# print("latent recon: ",F.mse_loss(X_final_latent, train_data, reduction='none').view(X_hat_latent.shape[0], -1).sum(1).mean())







z1 = log_std.exp() + mu
# Z = torch.cat((mu, z1), dim=0)
Z=mu
# pre_f Z
X, Z_mean, Z_var, V = pre_f(Z)
f,g = load_weights('vae_weights',pre_name='50')

# f,g=flatnet.train(X, n_iter=50, thres_recon=1e-5)
# pre_name = '50'
# torch.save(f.state_dict(), os.path.join('vae_weights', f'{pre_name}f.pth'))
# torch.save(g.state_dict(), os.path.join('vae_weights',  f'{pre_name}g.pth'))



latent = f(X)
# breakpoint()
latent_svd = dim_svd(latent)

# mu_latent,z1_latent=latent_svd.chunk(2,dim=0)
# latent_std=z1_latent-mu_latent
# latent_z=torch.randn_like(mu_latent) * latent_std + mu_latent
# X_hat_latent=g(latent_z)
# Z_hat_latent=in_F(X_hat_latent,V,Z_mean,Z_var)
# X_final_latent= model.decoder(Z_hat_latent)
# # breakpoint()
# print("latent recon: ",F.mse_loss(X_final_latent, train_data, reduction='none').view(X_hat_latent.shape[0], -1).sum(1).mean())



X_hat = g(latent)
Z_hat = in_F(X_hat,V,Z_mean,Z_var)

# loss between Z_hat,Z
print(F.mse_loss(Z,Z_hat, reduction='none').view(Z_hat.shape[0], -1).sum(1).mean())

# mu_hat, z1_hat = Z_hat.chunk(2, dim=0)
# std_hat = z1_hat-mu_hat
# std_hat = torch.abs(std_hat)
# log_std_hat = torch.log(std_hat)
# z = torch.randn_like(mu_hat) * std_hat + mu_hat
X_hat = model.decoder(Z_hat)
# loss_fn(train_data, X_hat, mu_hat, log_std_hat)
print("latent recon: ",F.mse_loss(X_hat, train_data, reduction='none').view(X_hat.shape[0], -1).sum(1).mean())
