from torchvision import datasets,transforms
import torch
import flatnet
import matplotlib.pyplot as plt
from flatnet.modules import flatnet_nn
import copy
import tqdm

transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor()
])

train = False
digit = 2
digit1 = 6

dataset = datasets.MNIST(root='./torch-dataset', train=True,
                        download=True,  transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=600000)
data,labels = next(iter(data_loader))
X_data = data[(labels == digit) | (labels == digit1)]
Z = X_data.reshape(X_data.shape[0], -1)
Z_mean = Z.mean(dim=0, keepdim=True)
Z_var=Z.norm(dim=1).max()+1e-8
Z = (Z - Z_mean) / Z_var
# U, S, Vt = torch.linalg.svd((Z - Z_mean)/ Z_var, full_matrices=False)
U, S, Vt = torch.linalg.svd(Z, full_matrices=False)
V = Vt.T
num_nonzero = 100
print(S[num_nonzero])
X= Z @ V[:,:num_nonzero]

if train:
    f,g=flatnet.train(X)
    torch.save(f.state_dict(), 'f.pth')
    torch.save(g.state_dict(), 'g.pth')
else:
    f = flatnet_nn.FlatteningNetwork()
    g = flatnet_nn.FlatteningNetwork()
    f_weights = torch.load('f.pth')
    g_weights = torch.load('g.pth')

    for i in range(f_weights['layer_count']):
        f_layer = flatnet_nn.FLayer(f_weights['network'][f'layer {i}.U'], f_weights['network'][f'layer {i}.z_mu_local'], f_weights['network'][f'layer {i}.gamma'], f_weights['network'][f'layer {i}.alpha'], f_weights['network'][f'layer {i}.z_mu'], f_weights['network'][f'layer {i}.z_norm'])
        g_layer = flatnet_nn.GLayer(g_weights['network'][f'layer {i}.U'], g_weights['network'][f'layer {i}.V'], g_weights['network'][f'layer {i}.z_mu_local'], g_weights['network'][f'layer {i}.x_c'], g_weights['network'][f'layer {i}.gamma'], g_weights['network'][f'layer {i}.alpha'], g_weights['network'][f'layer {i}.z_mu'], g_weights['network'][f'layer {i}.z_norm'])
        f.add_operation(f_layer)
        g.add_operation(g_layer)


original_g=copy.deepcopy(g)

f.requires_grad_(False)
for i in range(g.layer_count):
    for name,param in g.network[i].named_parameters():
        param.requires_grad_(False)
        if name == 'V':
            param.requires_grad_(True)

optimizer = torch.optim.Adam(g.parameters(), lr=0.01)

for i in tqdm.trange(1000):
    optimizer.zero_grad()
    X_hat = g(f(X))
    loss = torch.nn.functional.mse_loss(X_hat, X,reduction='mean')
    loss.backward()
    tqdm.tqdm.write(f'loss: {loss.item()}')
    optimizer.step()
    if loss < 5e-6:
        break

torch.save(g.state_dict(), 'g_trained.pth')
# g=copy.deepcopy(original_g)
# g.load_state_dict(torch.load('g_trained.pth'))
g.requires_grad_(False)

Z=f(X)
U, S, Vt = torch.linalg.svd(Z, full_matrices=False)
num_nonzero_Z = torch.sum(S > 1)
Z_svd = U[:,:num_nonzero_Z] @ torch.diag(S[:num_nonzero_Z])@Vt[:num_nonzero_Z,:]
X_hat_SVD=g(Z_svd)
X_hat=g(Z)

X_hat_SVD_original = original_g(Z_svd)
X_hat_original = original_g(Z)

# breakpoint()
Z_hat=X_hat@V[:,:num_nonzero].T*Z_var + Z_mean
Z_hat_original=X_hat_original@V[:,:num_nonzero].T*Z_var + Z_mean
Z_hat_SVD=X_hat_SVD@V[:,:num_nonzero].T*Z_var + Z_mean
Z_hat_SVD_original=X_hat_SVD_original@V[:,:num_nonzero].T*Z_var + Z_mean

X_data_hat = Z_hat.reshape(Z_hat.shape[0], 28, 28)
X_data_hat_SVD = Z_hat_SVD.reshape(Z_hat_SVD.shape[0], 28, 28)
X_data_hat_original = Z_hat_original.reshape(Z_hat_original.shape[0], 28, 28)
X_data_hat_SVD_original = Z_hat_SVD_original.reshape(Z_hat_SVD_original.shape[0], 28, 28)
X_data = X_data.detach().numpy()
X_data_hat = X_data_hat.detach().numpy()
X_data_hat_SVD = X_data_hat_SVD.detach().numpy()
X_data_hat_original = X_data_hat_original.detach().numpy()
X_data_hat_SVD_original = X_data_hat_SVD_original.detach().numpy()

D = X.shape[1]
# print(f'SVD of learned features: {torch.linalg.svd(Z - Z.mean(dim=0,keepdim=True))[1].shape}')
print(f'SVD of learned features: {num_nonzero_Z}')
print(f'Average reconstruction error: {(X-X_hat).norm(dim=1).mean() / (D ** 0.5)}')
print(f'Maximum reconstruction error: {(X-X_hat).norm(dim=1).max() / (D ** 0.5)}')

# plot the results
fig, ax = plt.subplots(4, 4, figsize=(10, 5))

for i in range(4):
    ax[i,0].imshow(X_data[i].reshape(28,28), cmap='gray')
    ax[i,0].set_title('Original')
    ax[i,1].imshow(X_data_hat[i].reshape(28,28), cmap='gray')
    ax[i,1].set_title('Reconstructed')
    ax[i,1].imshow(X_data_hat_SVD[i].reshape(28,28), cmap='gray')
    ax[i,1].set_title('Reconstructed_svd')
    # ax[i,3].imshow(X_data_hat_original[i].reshape(28,28), cmap='gray')
    # ax[i,3].set_title('Reconstructed_original')
    # ax[i,4].imshow(X_data_hat_SVD_original[i].reshape(28,28), cmap='gray')
    # ax[i,4].set_title('Reconstructed_svd_original')

    ax[i,2].imshow(X_data[i+5].reshape(28,28), cmap='gray')
    ax[i,2].set_title('Original')
    ax[i,1].imshow(X_data_hat[i+5].reshape(28,28), cmap='gray')
    ax[i,1].set_title('Reconstructed')
    ax[i,3].imshow(X_data_hat_SVD[i+5].reshape(28,28), cmap='gray')
    ax[i,3].set_title('Reconstructed_svd')

plt.show()