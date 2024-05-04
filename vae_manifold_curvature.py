from torch.utils import data
import train_vae
import torch
import torch.nn.functional as F
import os
import numpy as np
from torchvision import datasets, transforms
import flatnet
import matplotlib.pyplot as plt
from vae_manifold_train import torch_PCA

# from flatnet.modules import flatnet_nn
from mnist_test import dim_svd, load_weights,in_F,pre_f
import warnings
import geoopt
import tqdm
warnings.filterwarnings('ignore')

def loss_fn(x, x_hat, mu, log_std):
    recon_loss = F.mse_loss(x, x_hat, reduction='none').view(x.shape[0], -1).sum(1).mean()
    kl_loss = -log_std - 0.5 + (torch.exp(2 * log_std) + mu ** 2) * 0.5
    kl_loss = kl_loss.sum(1).mean()
    loss = recon_loss + kl_loss
    print(f'loss: {loss.item()}, recon_loss: {recon_loss.item()}, kl_loss: {kl_loss.item()}')

def enc_dec_f_g(f,g,pca):
    def enc_f(x):
        latent = f(x)
        z_d = pca.transform(latent)
        return z_d
    def dec_g(z_d):
        latent = pca.inverse_transform(z_d)
        x_hat = g(latent)
        return x_hat.squeeze(0)
    return enc_f,dec_g

def cal_curvature(Z,name,f,g,pca):
    enc_f,dec_g = enc_dec_f_g(f,g,pca)
    Z = Z.view(Z.shape[0], -1)
    N,D = Z.shape
    d = pca.n_components
    EDM_X = torch.cdist(Z,Z, p=2)
    edm_max = EDM_X.max()
    r = 0.1*edm_max
    curvature = torch.zeros(N)
    # print("mapping from R^{d} to R^{D}")
    print(f'Mapping from R^{d} to R^{D}')
    for i in tqdm.tqdm(range(N)):
        with torch.no_grad():
            z_c = Z[i]
            z_c_latent = enc_f(z_c)

        # def Jac_func(x):
        #     return torch.autograd.functional.jacobian(dec_g, x, create_graph=True).view(D,d)
        # H = torch.zeros(D,d)
        # for j in range(D):
        #     for k in range(d):
        #         # z_c_latent.requires_grad_(False)
        #         # z_c_latent[k].requires_grad_(True)
        #         z_c_latent=[z_c_latent[l].requires_grad_(True) if l==k else z_c_latent[l].requires_grad_(False) for l in range(d)]
        #         z_c_latent = tuple(z_c_latent)
        #         J = Jac_func(z_c_latent)
        #         breakpoint()
        #         J[j,k].backward(retain_graph=True)
        #         torch.autograd.grad(J[j,k],z_c_latent,create_graph=True)
        #         breakpoint()
        # form1 = J.T @ J # d*d
        # U,_,_ = torch.linalg.svd(J)
        # N = U[:,-1]
        # N = N / torch.linalg.norm(N)
        H = torch.zeros(D,d,d)
        for j in range(D):
            lambda_func = lambda xx:dec_g(xx)[j]
            H[j] = torch.autograd.functional.hessian(lambda_func, z_c_latent).view(d,d).detach()
            # only keep the diagonal
            H[j] = torch.diag(H[j])
        # H = torch.autograd.functional.jacobian(lambda x: torch.autograd.functional.jacobian(dec_g, x,create_graph=True).view(D,d), z_c_latent).view(D,d,d)
        # H = H.permute(1,2,0)
        # form2 = H @ N
        # # breakpoint()
        # curvature_matrix = torch.linalg.inv(form1) @ form2
        # eigenvalues, eigenvectors = torch.linalg.eig(curvature_matrix)
        # curvature[i] = eigenvalues.real.max()
        curvature[i] = torch.linalg.norm(H,dim=(1,2)).max()
        # breakpoint()
        # print(f'{i} curvature: {curvature[i]}')

        # write curvature to tqdm
        tqdm.tqdm.write(f'{i} curvature: {curvature[i]}')
    print(f'{name} curvature: {curvature}')
    print(f'mean_curv_x: {curvature.mean()}, std_curv_x: {curvature.std()}')
    return curvature

if __name__ == '__main__':
    weight_folder = 'vae_weights_mnist'
    pre_name = '50'
    pairs = 10
    interpolation_steps = 10
    device = torch.device('cpu')

    # dataset
    transform = transforms.Compose([transforms.ToTensor()])
    transform_to28 = transforms.Resize(28)
    transform_to32 = transforms.Resize(32)
    test_data = datasets.MNIST(root='./torch-dataset', train=True,download=True,  transform=transform)
    test_data = np.array(transform_to32(test_data.data))
    test_data = test_data[:,:,:,None]
    test_data = (np.transpose(test_data, (0, 3, 1, 2)) / 255.).astype('float32')
    test_loader = data.DataLoader(test_data, batch_size=20,generator=torch.Generator(device=device),shuffle=True)
    test_data = next(iter(test_loader))

    # model
    model=train_vae.ConvVAE((1, 32, 32), 32).to(device)
    model.load_state_dict(torch.load(os.path.join(weight_folder, 'vae_model.pth'), map_location=device))
    model.eval()
    model.requires_grad_(False)

    # # evaluate the VAE
    # x = 2 * test_data - 1
    # z, _ = model.encoder(x)
    # x_recon = torch.clamp(model.decoder(z), -1, 1)
    # reconstructions = x_recon.view(-1, 1, 32, 32) * 0.5 + 0.5
    # reconstructions = reconstructions.permute(0, 2, 3, 1).cpu() * 255
    # reconstructions = reconstructions.squeeze(3)
        
    # evaluate the flatnet
    f,g = load_weights(weight_folder,pre_name=pre_name)
    pca = torch_PCA.load(os.path.join(weight_folder, f'{pre_name}pca.pth'))

    with torch.no_grad():
        test_data = test_data.to(device)
        test_data = 2*test_data - 1
        Z, _ = model.encoder(test_data)
        X, Z_mean, Z_var, V = pre_f(Z)
        latent = f(X)
        latent,d = dim_svd(latent)
        X_xx=torch.clamp(model.decoder(in_F(g(latent),V,Z_mean,Z_var)), -1, 1)
    cal_curvature(X, "recon_manifold",f,g,pca)
    # with torch.no_grad():
    #     X_hat = g(latent)
    #     Z_hat = in_F(X_hat,V,Z_mean,Z_var)
    #     test_recon = torch.clamp(model.decoder(Z_hat), -1, 1)
    # reconstructions = test_recon.view(-1, 1, 32, 32) * 0.5 + 0.5
    # reconstructions = reconstructions.permute(0, 2, 3, 1).cpu() * 255
    # reconstructions = reconstructions.squeeze(3)
