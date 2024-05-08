import argparse
import flatnet
import utils
from flatnet.modules import flatnet_nn
import tqdm
import torch

class ManifoldCurvature:
    def __init__(self):
        self.d = None
        self.D = None
        self.func_f = None
        self.func_g = None

    # def from_fg(self, f, g, pca):
    #     def enc_f(self,x):
    #         latent = f(x)
    #         z_d = pca.transform(latent)
    #         return z_d
    
    #     def dec_g(self,z_d):
    #         latent = pca.inverse_transform(z_d)
    #         x_hat = g(latent)
    #         return x_hat.squeeze(0)
        
    #     return enc_f, dec_g
    
    def fit(self, X,latent_dim=None, svd_threshold=1, min_dim=1, n_iter=150):       
        X= X.view(X.shape[0], -1)
        _, self.D = X.shape
        f, g = flatnet.train(X, n_iter=n_iter)
        pca = utils.torch_PCA()
        if latent_dim is not None:
            self.d = latent_dim
            pca.fit(f(X), latent_dim=latent_dim)
        else:
            self.d = pca.fit(f(X), svd_threshold=svd_threshold, min_dim=min_dim)
        def enc_f(x):
            latent = f(x)
            z_d = pca.transform(latent)
            return z_d
    
        def dec_g(z_d):
            latent = pca.inverse_transform(z_d)
            x_hat = g(latent)
            return x_hat.squeeze(0)
        self.func_f = enc_f
        self.func_g = dec_g
    
    def tangent_space(self, z_c):
        z_c = z_c.view(1, -1)
        assert z_c.shape[1] == self.D
        with torch.no_grad():
            z_c_latent = self.func_f(z_c)
        return torch.autograd.functional.jacobian(self.func_g, z_c_latent, create_graph=False).view(self.D, self.d)   

    def hessian(self, z_c):
        z_c = z_c.view(1, -1)
        assert z_c.shape[1] == self.D
        with torch.no_grad():
            z_c_latent = self.func_f(z_c)
        H = torch.zeros(self.D, self.d, self.d)
        for j in range(self.D):
            lambda_func = lambda xx: self.func_g(xx)[j]
            H[j] = torch.autograd.functional.hessian(lambda_func, z_c_latent, create_graph=False).view(self.d, self.d).detach()
            H[j] = torch.diag(H[j])
        return H

    def curvature(self, X):
        X = X.view(X.shape[0], -1)
        assert X.shape[1] == self.D
        curvature = torch.zeros(X.shape[0])
        for i in tqdm.tqdm(range(X.shape[0])):
            z_c = X[i]
            H = self.hessian(z_c)
            trace = torch.zeros(self.D)
            # for j in range(self.D):
            #     trace[j] = torch.sum(H[j])
            # curvature[i] = trace.max()
            curvature[i] = torch.sum(H)
        return curvature      
        
