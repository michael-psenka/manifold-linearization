import torch
import os
from flatnet.modules import flatnet_nn


class torch_PCA:
    def __init__(self):
        self.n_components = None
        self.components_ = None
        self.mean_ = None

    def fit(self, X,latent_dim=None, svd_threshold=1, min_dim=1):
        self.mean_ = X.mean(0)
        X = X - self.mean_
        U, S, V = torch.linalg.svd(X, full_matrices=False)
        if latent_dim is not None:
            self.n_components = latent_dim
        else:
            self.n_components = torch.sum(S > svd_threshold)
            if self.n_components<min_dim:
                self.n_components = min_dim
        self.components_ = V[:, :self.n_components]
        return self.n_components
        

    def transform(self, X):
        return (X - self.mean_) @ self.components_

    def inverse_transform(self, X):
        return X @ self.components_.T + self.mean_
    
    def save(self, path):
        torch.save(self, path)

    @staticmethod
    def load(path):
        return torch.load(path)
    


def dim_svd(Z,threshold=1,min_dim=1):
    U_Z, S_Z, Vt_Z = torch.linalg.svd(Z, full_matrices=False)
    num_nonzero_Z = torch.sum(S_Z > threshold)
    if num_nonzero_Z<min_dim:
        num_nonzero_Z = min_dim
    Z_svd = U_Z[:,:num_nonzero_Z] @ torch.diag(S_Z[:num_nonzero_Z])@Vt_Z[:num_nonzero_Z,:]
    print(f'SVD of learned features: {num_nonzero_Z}')
    return Z_svd, num_nonzero_Z