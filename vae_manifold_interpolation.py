from torch.utils import data
import train_vae
import torch
import torch.nn.functional as F
import os
import numpy as np
from torchvision import datasets, transforms
import flatnet
import matplotlib.pyplot as plt

# from flatnet.modules import flatnet_nn
from mnist_test import dim_svd, load_weights,in_F,pre_f
import warnings

warnings.filterwarnings('ignore')

def loss_fn(x, x_hat, mu, log_std):
    recon_loss = F.mse_loss(x, x_hat, reduction='none').view(x.shape[0], -1).sum(1).mean()
    kl_loss = -log_std - 0.5 + (torch.exp(2 * log_std) + mu ** 2) * 0.5
    kl_loss = kl_loss.sum(1).mean()
    loss = recon_loss + kl_loss
    print(f'loss: {loss.item()}, recon_loss: {recon_loss.item()}, kl_loss: {kl_loss.item()}')



if __name__ == '__main__':
    weight_folder = 'vae_weights_mnist'
    folder = 'vae_mnist_interpolation'
    pre_name = '50'
    pairs = 10
    interpolation_steps = 10
    device = torch.device('cpu')

    os.makedirs(folder, exist_ok=True)
    # dataset
    transform = transforms.Compose([transforms.ToTensor()])
    transform_to28 = transforms.Resize(28)
    transform_to32 = transforms.Resize(32)
    test_data = datasets.MNIST(root='./torch-dataset', train=False,download=True,  transform=transform)
    test_data = np.array(transform_to32(test_data.data))
    test_data = test_data[:,:,:,None]
    test_data = (np.transpose(test_data, (0, 3, 1, 2)) / 255.).astype('float32')
    test_loader = data.DataLoader(test_data, batch_size=2*pairs,generator=torch.Generator(device=device),shuffle=True)
    test_data = next(iter(test_loader))

    # model
    model=train_vae.ConvVAE((1, 32, 32), 32).to(device)
    model.load_state_dict(torch.load(os.path.join(weight_folder, 'vae_model.pth'), map_location=device))
    model.eval()
    model.requires_grad_(False)

    # evaluate the VAE
    x = 2 * test_data - 1
    z, _ = model.encoder(x)
    z1 = z[:pairs]
    z2 = z[pairs:]
    z = z1
    for i in range(interpolation_steps-1):
        z = torch.cat((z, z1 + (z2 - z1) * (i+1) / interpolation_steps), dim=0)
    z = torch.cat((z, z2), dim=0)
    x_recon = torch.clamp(model.decoder(z), -1, 1)
    reconstructions = x_recon.view(-1, 1, 32, 32) * 0.5 + 0.5
    reconstructions = reconstructions.permute(0, 2, 3, 1).cpu() * 255
    reconstructions = reconstructions.squeeze(3)

    fig, axes = plt.subplots(pairs,interpolation_steps+1, figsize=(int(1.2*interpolation_steps),int(1.2*pairs)))
    for i in range(pairs):
        for j in range(interpolation_steps+1):
            ax = axes[i,j]
            ax.imshow(reconstructions[j*interpolation_steps+i].numpy().astype('uint8'))
            ax.axis('off')
    plt.savefig(os.path.join(folder,'vae_interpolations.png'))

        
    # evaluate the flatnet
    f,g = load_weights(weight_folder,pre_name=pre_name)
    with torch.no_grad():
        test_data = test_data.to(device)
        test_data = 2*test_data - 1
        Z, _ = model.encoder(test_data)
        X, Z_mean, Z_var, V = pre_f(Z)
        latent = f(X)
        latent = dim_svd(latent)
        latent1 = latent[:pairs]
        latent2 = latent[pairs:]
        latent = latent1
        for i in range(interpolation_steps-1):
            latent = torch.cat((latent, latent1 + (latent2 - latent1) * (i+1) / interpolation_steps), dim=0)
        latent = torch.cat((latent, latent2), dim=0)
        X_hat = g(latent)
        Z_hat = in_F(X_hat,V,Z_mean,Z_var)
        test_recon = torch.clamp(model.decoder(Z_hat), -1, 1)
    reconstructions = test_recon.view(-1, 1, 32, 32) * 0.5 + 0.5
    reconstructions = reconstructions.permute(0, 2, 3, 1).cpu() * 255
    reconstructions = reconstructions.squeeze(3)

    fig, axes = plt.subplots(pairs,interpolation_steps+1, figsize=(int(1.2*interpolation_steps),int(1.2*pairs)))
    for i in range(pairs):
        for j in range(interpolation_steps+1):
            ax = axes[i,j]
            ax.imshow(reconstructions[j*interpolation_steps+i].numpy().astype('uint8'))
            ax.axis('off')
    plt.savefig(os.path.join(folder,'flatnet_interpolations.png'))
