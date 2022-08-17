import torch

from pykeops.torch import Genred

def flattening_deformation(X, K, eps=0.1, Cmax = 100, a1=0.1, a2=0.1):
    n, d = X.shape

    # Step 1. Compute Euclidean distances and KNN
    #could replace with Gramian matrix
    Xi = X[:, None, :]
    Xj = X[None, :, :]
    Pij = Xi - Xj
    Dij = ((Xi - Xj) ** 2).sum(-1)
    knn = Dij.topk(K+1, largest=False)


    #Compute soft neighborhoods
    Di = knn.values[:, 1:]
    Dmin = Di[:, 0].unsqueeze(-1)
    NDij = Dmin/Di
    Nj = knn.indices[:, 1:]


    mask = torch.zeros(n, n).scatter_(1, Nj, NDij)
    Xc = X
    C = 0
    while C < Cmax:
        Xci = Xc[:, None, :]
        Xcj = Xc[None, :, :]
        Dcij = ((Xci - Xcj) ** 2).sum(-1)

        Vr = torch.nan_to_num(Pij * ((1 - mask)/Dcij).unsqueeze(-1))
        Ve = torch.nan_to_num(Pij * (mask * (Dij - Dcij)/Dcij).unsqueeze(-1))

        V = a1 * torch.sum(Vr, dim=1) + a2 * torch.sum(Ve, dim=1)
        if torch.norm(V, p='inf') < eps:
            return Xc
        Xc = Xc + V
        C = C+1

    return Xc
    

N = 100
D = 10
X = torch.rand(N, D)
flattening_deformation(X, 10, 0.1, 15, .1, .1)
