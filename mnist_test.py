from torchvision import datasets,transforms
import torch
import flatnet
import matplotlib.pyplot as plt
from flatnet.modules import flatnet_nn
import copy
import tqdm
import numpy as np
import os
from sklearn.decomposition import PCA

iters = 150
iters2 = 50
train1 = True
train2 = True

digit = [0,1,2,3,4,5,6,7,8,9]
num_nonzero = 100
svd_Z_threshold = 1
use_svd = False
with_labels = False
batch_size = 256
weight_folder='0-9_weights_batch_1e-5'
num_per_class = 1500


if use_svd:
    weight_folder += '_svd'


os.makedirs(weight_folder, exist_ok=True)
digits = [0,1,2,3,4,5,6,7,8,9]

def pre_f(X_data):
    Z=X_data.reshape(X_data.shape[0], -1)
    Z_mean = Z.mean(dim=0, keepdim=True)
    Z_var=Z.norm(dim=1).max()+1e-8
    Z = (Z - Z_mean) / Z_var
    if use_svd:
        U, S, Vt = torch.linalg.svd(Z, full_matrices=False)
        V = Vt.T
        print(S[num_nonzero])
        X= Z @ V[:,:num_nonzero]
    else:
        X = Z
        V = None
    if with_labels:
        one_hot_labels = torch.zeros(X_data.shape[0], 10)
        for i in range(10):
            one_hot_labels[i*num_per_class:(i+1)*num_per_class, i] = 1
        X = torch.cat((X, one_hot_labels), dim=1)
        # concatenate one-hot labels to the data

    return X, Z_mean, Z_var, V

def pre_mean_var(X_data,Z_mean,Z_var,V):
    Z = X_data.reshape(X_data.shape[0], -1)
    Z = (Z - Z_mean) / Z_var
    if use_svd:
        X = Z @ V[:,:num_nonzero]
    else:
        X = Z
    return X

def pca_latent(Z,figname):
    pca = PCA(n_components=3)
    latent_2d = pca.fit_transform(Z.cpu().detach().numpy())

    # 3D plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(latent_2d[:, 0], latent_2d[:, 1], latent_2d[:, 2], alpha=0.3, c=labels.cpu().detach().numpy(), cmap='tab10')


    # plt.figure(figsize=(8, 6))
    # plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.3, c=labels.cpu().detach().numpy(), cmap='tab10')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA')
    plt.grid(True)
    plt.savefig(os.path.join(weight_folder, figname))
    plt.show()
    # clear everything plotted
    plt.close()



    return latent_2d

def dim_svd(Z,threshold=svd_Z_threshold):
    U_Z, S_Z, Vt_Z = torch.linalg.svd(Z, full_matrices=False)
    num_nonzero_Z = torch.sum(S_Z > threshold)
    if num_nonzero_Z<7:
        num_nonzero_Z = 7
    Z_svd = U_Z[:,:num_nonzero_Z] @ torch.diag(S_Z[:num_nonzero_Z])@Vt_Z[:num_nonzero_Z,:]
    print(f'SVD of learned features: {num_nonzero_Z}')
    return Z_svd, num_nonzero_Z

def in_F(X_hat, V, Z_mean, Z_var):
    if with_labels:
        X_hat = X_hat[:,:-10]
    if use_svd:
        X_data_hat = X_hat @ V[:,:num_nonzero].T * Z_var + Z_mean
    else:
        X_data_hat = X_hat * Z_var + Z_mean
    # X_data_hat = X_data_hat.reshape(X_data_hat.shape[0], 28, 28)
    return X_data_hat

def load_weights(weight_folder,pre_name=''):
    f = flatnet_nn.FlatteningNetwork()
    g = flatnet_nn.FlatteningNetwork()
    f_weights = torch.load(os.path.join(weight_folder, f'{pre_name}f.pth'))
    g_weights = torch.load(os.path.join(weight_folder, f'{pre_name}g.pth'))

    for i in range(f_weights['layer_count']):
        f_layer = flatnet_nn.FLayer(f_weights['network'][f'layer {i}.U'], f_weights['network'][f'layer {i}.z_mu_local'], f_weights['network'][f'layer {i}.gamma'], f_weights['network'][f'layer {i}.alpha'], f_weights['network'][f'layer {i}.z_mu'], f_weights['network'][f'layer {i}.z_norm'])
        g_layer = flatnet_nn.GLayer(g_weights['network'][f'layer {i}.U'], g_weights['network'][f'layer {i}.V'], g_weights['network'][f'layer {i}.z_mu_local'], g_weights['network'][f'layer {i}.x_c'], g_weights['network'][f'layer {i}.gamma'], g_weights['network'][f'layer {i}.alpha'], g_weights['network'][f'layer {i}.z_mu'], g_weights['network'][f'layer {i}.z_norm'])
        f.add_operation(f_layer)
        g.add_operation(g_layer)
    return f, g



if __name__ == '__main__':
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(root='./torch-dataset', train=True,download=True,  transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=600000)
    data,all_labels = next(iter(data_loader))
    data = data.cuda()
    if type(digit) == int:
        X_data = data[all_labels == digit]
        assert digit in digits
    else:
        for i in digit:
            assert i in digits
        X_data = data[all_labels == digit[0]][:num_per_class]
        labels = all_labels[all_labels == digit[0]][:num_per_class]
        for i in digit[1:]:
            X_data = torch.cat((X_data, data[all_labels == i][:num_per_class]), dim=0)
            labels = torch.cat((labels, all_labels[all_labels == i][:num_per_class]), dim=0)
    X, Z_mean, Z_var, V = pre_f(X_data)
    if train1:
        f,g=flatnet.train(X, iters,thres_recon=1e-5)
        torch.save(f.state_dict(), os.path.join(weight_folder, 'f.pth'))
        torch.save(g.state_dict(), os.path.join(weight_folder, 'g.pth'))
    else:
        f,g = load_weights(weight_folder)
    original_g=copy.deepcopy(g)
    pca_latent(X,'original_pca.png')
    Z_all = f(X)
    pca_latent(Z_all,'latent_pca.png')

    f.requires_grad_(False)
    if train2:
        g.requires_grad_(True)
        for i in range(g.layer_count):
            for name,param in g.network[i].named_parameters():
                param.requires_grad_(False)
                if name == 'V':
                    param.requires_grad_(True)

        optimizer = torch.optim.Adam(g.parameters(), lr=0.001)

        for i in tqdm.trange(iters2):
            total_loss = 0
            for batch in torch.utils.data.DataLoader(X, batch_size=batch_size, shuffle=True,generator=torch.Generator(device='cuda')):
                optimizer.zero_grad()
                Z = f(batch)
                X_hat = g(Z)
                loss = torch.nn.functional.mse_loss(X_hat, batch,reduction='mean')
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            tqdm.tqdm.write(f'loss: {total_loss}')
            if total_loss < 1e-5:
                break
            assert not loss.isnan()
        torch.save(g.state_dict(), os.path.join(weight_folder, 'g_trained.pth'))
    else:
        g.load_state_dict(torch.load(os.path.join(weight_folder, 'g_trained.pth')))
    
    print('Starting evaluation')
    g.requires_grad_(False)
    g.eval()
    f.eval()
    original_g.eval()
    
    for i in range(len(digit)):
        X_i = data[all_labels == digit[i]][:2]
        if i == 0:
            X_0 = X_i
        else:
            X_0 = torch.cat((X_0, X_i), dim=0)
    X_data = X_0.cuda()
    X = pre_mean_var(X_data, Z_mean, Z_var, V)
    Z=f(X)
    X_hat=g(Z)
    Z_svd = dim_svd(Z)
    X_hat_SVD=g(Z_svd)
    X_hat_original = original_g(Z)
    X_hat_SVD_original = original_g(Z_svd)

    X_SVD = in_F(X, V, Z_mean, Z_var)
    X_data_hat = in_F(X_hat, V, Z_mean, Z_var)
    X_data_hat_SVD = in_F(X_hat_SVD, V, Z_mean, Z_var)
    X_data_hat_original = in_F(X_hat_original, V, Z_mean, Z_var)
    X_data_hat_SVD_original = in_F(X_hat_SVD_original, V, Z_mean, Z_var)
    X_data = X_data.cpu().detach().numpy()

    D = X.shape[1]
    print(f'Average reconstruction error: {(X-X_hat).norm(dim=1).mean() / (D ** 0.5)}')
    print(f'Maximum reconstruction error: {(X-X_hat).norm(dim=1).max() / (D ** 0.5)}')

# plot the results
    fig, ax = plt.subplots(len(digit), 6, figsize=(8, 5))

    for i in range(len(digit)):
        x=int(2*i)
        # if i<4:
        #     x = int(0)
        # else:
        #     x = int(1/2*len(X_data))
        ax[i,0].imshow(X_data[x].reshape(28,28), cmap='gray')
        ax[i,0].set_title('Original')
        # ax[i,1].imshow(X_data_hat[x].reshape(28,28), cmap='gray')
        # ax[i,1].set_title('Re: {:.2f}'.format(np.linalg.norm(X_data[x]-X_data_hat[x])))
        ax[i,1].imshow(X_data_hat_SVD[x].reshape(28,28), cmap='gray')
        ax[i,1].set_title('Re_svd: {:.2f}'.format(np.linalg.norm(X_data[x]-X_data_hat_SVD[x])))
        # ax[i,3].imshow(X_data_hat_original[x].reshape(28,28), cmap='gray')
        # ax[i,3].set_title('Re_ori: {:.2f}'.format(np.linalg.norm(X_data[x]-X_data_hat_original[x])))
        ax[i,2].imshow(X_data_hat_SVD_original[x].reshape(28,28), cmap='gray')
        ax[i,2].set_title('Re_svd_ori: {:.2f}'.format(np.linalg.norm(X_data[x]-X_data_hat_SVD_original[x])))
        # ax[i,5].imshow(X_SVD[i].reshape(28,28), cmap='gray')
        # ax[i,5].set_title('Original_svd: {:.2f}'.format(np.linalg.norm(X_data[x]-X_SVD[i])))
        x=int(2*i+1)
        ax[i,3].imshow(X_data[x].reshape(28,28), cmap='gray')
        ax[i,3].set_title('Original')
        ax[i,4].imshow(X_data_hat_SVD[x].reshape(28,28), cmap='gray')
        ax[i,4].set_title('Re_svd: {:.2f}'.format(np.linalg.norm(X_data[x]-X_data_hat_SVD[x])))
        ax[i,5].imshow(X_data_hat_SVD_original[x].reshape(28,28), cmap='gray')
        ax[i,5].set_title('Re_svd_ori: {:.2f}'.format(np.linalg.norm(X_data[x]-X_data_hat_SVD_original[x])))
    fig.savefig(os.path.join(weight_folder, 'results.png'))
    # plt.show()