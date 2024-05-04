from torch.utils import data
import train_vae
import torch
import torch.nn.functional as F
import os
import numpy as np
from torchvision import datasets, transforms
import flatnet
import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# import pickle

from mnist_test import dim_svd, load_weights,in_F,pre_f
import warnings

warnings.filterwarnings('ignore')

def loss_fn(x, x_hat, mu, log_std):
    recon_loss = F.mse_loss(x, x_hat, reduction='none').view(x.shape[0], -1).sum(1).mean()
    kl_loss = -log_std - 0.5 + (torch.exp(2 * log_std) + mu ** 2) * 0.5
    kl_loss = kl_loss.sum(1).mean()
    loss = recon_loss + kl_loss
    print(f'loss: {loss.item()}, recon_loss: {recon_loss.item()}, kl_loss: {kl_loss.item()}')

# pca = PCA(n_components=int(d))
#         pca.fit(f(X).cpu().detach().numpy())
#         with open(os.path.join(folder, f'{pre_name}pca.pkl'), 'wb') as f:
#             # pickle.dump(pca, f)

# fit, transform, inverse_transform, implement PCA functions in torch
class torch_PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None

    def fit(self, X):
        self.mean_ = X.mean(0)
        X = X - self.mean_
        U, S, V = torch.linalg.svd(X, full_matrices=False)
        self.components_ = V[:, :self.n_components]
        

    def transform(self, X):
        return (X - self.mean_) @ self.components_

    def inverse_transform(self, X):
        return X @ self.components_.T + self.mean_
    
    def save(self, path):
        torch.save(self, path)

    @staticmethod
    def load(path):
        return torch.load(path)
    

if __name__ == '__main__':
    train_manifold = True
    folder = 'vae_weights_mnist'
    pre_name = '50'
    device = torch.device('cpu')

    # dataset
    transform = transforms.Compose([transforms.ToTensor()])
    transform_to28 = transforms.Resize(28)
    transform_to32 = transforms.Resize(32)
    train_data = datasets.MNIST(root='./torch-dataset', train=True,download=True,  transform=transform)
    test_data = datasets.MNIST(root='./torch-dataset', train=False,download=True,  transform=transform)

    train_data = np.array(transform_to32(train_data.data))
    test_data = np.array(transform_to32(test_data.data))

    train_data = train_data[:,:,:,None]
    test_data = test_data[:,:,:,None]

    train_data = (np.transpose(train_data, (0, 3, 1, 2)) / 255.).astype('float32')
    test_data = (np.transpose(test_data, (0, 3, 1, 2)) / 255.).astype('float32')    

    train_loader = data.DataLoader(train_data, batch_size=2000, shuffle=True,generator=torch.Generator(device=device))
    test_loader = data.DataLoader(test_data, batch_size=2000,generator=torch.Generator(device=device))

    train_data = next(iter(train_loader))
    test_data = next(iter(test_loader))

    # model
    model=train_vae.ConvVAE((1, 32, 32), 32).to(device)
    model.load_state_dict(torch.load(os.path.join(folder, 'vae_model.pth'), map_location=device))
    model.eval()
    model.requires_grad_(False)

    # evaluate the VAE
    x = 2 * test_data - 1
    z, _ = model.encoder(x)
    x_recon = torch.clamp(model.decoder(z), -1, 1)
    print("VAE recon: ",F.mse_loss(x, x_recon, reduction='none').view(x.shape[0], -1).sum(1).mean())
    reconstructions = torch.stack((x[:10], x_recon[:10]), dim=1).view(-1, 1, 32, 32) * 0.5 + 0.5
    reconstructions = reconstructions.permute(0, 2, 3, 1).cpu() * 255
    reconstructions = reconstructions.squeeze(3)

    fig, axes = plt.subplots(5,4, figsize=(10,8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(reconstructions[i].numpy().astype('uint8'))
        ax.set_title('Original' if i % 2 == 0 else 'VAE: '+ str(int(torch.sqrt(F.mse_loss(reconstructions[i-1],reconstructions[i],reduction='mean')))))
        ax.axis('off')
    plt.savefig(os.path.join(folder,'vae_reconstructions.png'))

    # train the flatnet
    if train_manifold:
        train_data = train_data.to(device)
        train_data = 2 * train_data - 1
        mu, log_std = model.encoder(train_data)
        Z= mu
        X, Z_mean, Z_var, V = pre_f(Z)
        # f,g=flatnet.train(X, n_iter=50, thres_recon=1e-5)
        f,g = load_weights(folder,pre_name=pre_name)
        _, d=dim_svd(f(X))
        pca = torch_PCA(d)
        pca.fit(f(X))
        pca.save(os.path.join(folder, f'{pre_name}pca.pth'))              

        torch.save(f.state_dict(), os.path.join(folder, f'{pre_name}f.pth'))
        torch.save(g.state_dict(), os.path.join(folder,  f'{pre_name}g.pth'))
    else:
        f,g = load_weights(folder,pre_name=pre_name)

    # evaluate the flatnet
    with torch.no_grad():
        test_data = test_data.to(device)
        test_data = 2*test_data - 1
        Z, _ = model.encoder(test_data)
        X, Z_mean, Z_var, V = pre_f(Z)
        latent = f(X)
        latent_svd,_ = dim_svd(latent)
        X_hat = g(latent)
        X_hat_svd = g(latent_svd)
        Z_hat = in_F(X_hat,V,Z_mean,Z_var)
        Z_hat_svd = in_F(X_hat_svd,V,Z_mean,Z_var)
        test_recon = torch.clamp(model.decoder(Z_hat), -1, 1)
        test_recon_svd = torch.clamp(model.decoder(Z_hat_svd), -1, 1)
        print("flatnet recon: ",F.mse_loss(test_data,test_recon, reduction='none').view(test_recon.shape[0], -1).sum(1).mean())
        print("flatnet recon svd: ",F.mse_loss(test_data,test_recon_svd, reduction='none').view(test_recon_svd.shape[0], -1).sum(1).mean())
    reconstructions = torch.stack((test_data[:10], test_recon[:10], test_recon_svd[:10]), dim=1).view(-1, 1, 32, 32) * 0.5 + 0.5
    reconstructions = reconstructions.permute(0, 2, 3, 1).cpu() * 255
    reconstructions = reconstructions.squeeze(3)

    fig, axes = plt.subplots(5,6, figsize=(10,12))
    for i, ax in enumerate(axes.flat):
        ax.imshow(reconstructions[i].numpy().astype('uint8'))

        if i %3 == 0:
            ax.set_title('Original')
        elif i %3 == 1:
            ax.set_title('Flatnet: '+ str(int(torch.sqrt(F.mse_loss(reconstructions[i-1],reconstructions[i],reduction='mean')).item())))
        else:
            ax.set_title('Flatnet SVD: '+str(int(torch.sqrt(F.mse_loss(reconstructions[i-2],reconstructions[i],reduction='mean')).item())))
        ax.axis('off')
    plt.savefig(os.path.join(folder,'flatnet_reconstructions.png'))
