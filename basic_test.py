import torch
import flatnet
import matplotlib.pyplot as plt
import copy
import tqdm
from flatnet.modules import flatnet_nn

# create sine wave dataset
t = torch.linspace(0, 2*torch.pi, 50)
y = torch.sin(t)

# format dataset of N points of dimension D as (N, D) matrix
X = torch.stack([t, y], dim=1)
# add noise
X = X + 0.02*torch.randn(*X.shape)

# normalize data
X = X - X.mean(dim=0)
X = X / X.norm(dim=1).max()

# train encoder f and decoder g, then save a gif of the
# manifold evolution through the constructed layers of f

train = True

if train:
    f, g = flatnet.train(X, n_iter=100, save_gif=True)
    torch.save(f.state_dict(), 'f_basic.pth')
    torch.save(g.state_dict(), 'g_basic.pth')
else:
    # load the trained models
    f = flatnet_nn.FlatteningNetwork()
    g = flatnet_nn.FlatteningNetwork()
    f_weights = torch.load('f_basic.pth')
    g_weights = torch.load('g_basic.pth')

    for i in range(f_weights['layer_count']):
        f_layer = flatnet_nn.FLayer(f_weights['network'][f'layer {i}.U'], f_weights['network'][f'layer {i}.z_mu_local'], f_weights['network'][f'layer {i}.gamma'], f_weights['network'][f'layer {i}.alpha'], f_weights['network'][f'layer {i}.z_mu'], f_weights['network'][f'layer {i}.z_norm'])
        g_layer = flatnet_nn.GLayer(g_weights['network'][f'layer {i}.U'], g_weights['network'][f'layer {i}.V'], g_weights['network'][f'layer {i}.z_mu_local'], g_weights['network'][f'layer {i}.x_c'], g_weights['network'][f'layer {i}.gamma'], g_weights['network'][f'layer {i}.alpha'], g_weights['network'][f'layer {i}.z_mu'], g_weights['network'][f'layer {i}.z_norm'])
        f.add_operation(f_layer)
        g.add_operation(g_layer)


# original_g=copy.deepcopy(g)

# f.requires_grad_(False)
# for i in range(g.layer_count):
#     for name,param in g.network[i].named_parameters():
#         param.requires_grad_(False)
#         if name == 'V':
#             param.requires_grad_(True)

# optimizer = torch.optim.Adam(g.parameters(), lr=0.01)

# for i in tqdm.tqdm(range(100)):
#     optimizer.zero_grad()
#     X_hat = g(f(X))
#     # breakpoint()
#     loss = torch.nn.functional.mse_loss(X_hat, X,reduction='mean')
#     loss.backward()
#     # add loss to tqdm
#     tqdm.tqdm.write(f'loss: {loss.item()}')
#     optimizer.step()

# g.requires_grad_(False)
#save
torch.save(g.state_dict(), 'g_trained.pth')

# reconstruct the data
Z = f(X)
Z = Z.detach().numpy()
# X_hat_original = original_g(Z)
X_hat = g(f(X)).detach().numpy()

# X_hat_original = X_hat_original.detach().numpy()
# plot the results
fig, ax = plt.subplots(1, 2)
ax[0].plot(X[:, 0], X[:, 1], 'o',color='blue')
ax[0].plot(X_hat[:, 0], X_hat[:, 1], 'o',color='orange')
# ax[0].plot(X_hat_original[:, 0], X_hat_original[:, 1], 'o',color='green')
ax[0].set_title('Original and Reconstructed Data')
ax[1].plot(Z[:, 0], Z[:, 1], 'o')
ax[1].set_title('Latent Space')
plt.show()


