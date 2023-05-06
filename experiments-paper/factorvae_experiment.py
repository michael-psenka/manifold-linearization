from itertools import chain
import os

import sys
sys.path.append('../')

import torch
import pytorch_lightning as pl

from tools.gp_manifold_generator import sample_points
from models.vae import FactorVAE

N = 6000
D = 100
d = 10
gamma = 30
L = 5
lr = 1e-3
batch_size = N // 2
num_epochs = 1000


X_filename = f"factorvae_experiment/X_N{N}_D{D}_d{d}.pt"
P_filename = f"factorvae_experiment/P_N{N}_D{D}_d{d}.pt"
model_filename = f"factorvae_experiment/model(COMPRESSED FEATURES)_N{N}_D{D}_d{d}_gam{gamma}_L{5}_lr{lr}_bs{batch_size}_e{num_epochs}.ckpt"

if not (os.path.exists(X_filename) and os.path.exists(P_filename)):
    X, P, d = sample_points(N, D, d, [1.0 for _ in range(D)])
    torch.save(X, X_filename)
    torch.save(P, P_filename)
X = torch.load(X_filename)
P = torch.load(P_filename)
print('done generating!')
if not os.path.exists(model_filename):
    trainer = pl.Trainer(max_epochs=num_epochs)
    dataloader = torch.utils.data.DataLoader(X, batch_size, drop_last=False)
    model = FactorVAE(gamma=gamma, d_x=X.shape[1], d_z=d, d_latent=d, n_layers=L, lr=lr)
    trainer.fit(model, train_dataloaders=dataloader)
    torch.save(model.state_dict(), model_filename)
model = FactorVAE(gamma=gamma, d_x=X.shape[1], d_z=X.shape[1], d_latent=X.shape[1], n_layers=L, lr=lr)
model.load_state_dict(torch.load(model_filename))
