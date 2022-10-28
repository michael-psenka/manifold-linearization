import torch
from torchdiffeq import odeint

# Class representing a smooth manifold from which we draw data points
# manifold represented as output of a single smooth function

# Adapted from the accompanying code for the following paper:
'''
Zenke, F., and Vogels, T.P. (2021). The Remarkable Robustness of Surrogate Gradient Learning for 
Instilling Complex Function in Spiking Neural Networks. Neural Computation 1–27.
'''
class Manifold:
  def __init__(self, D, d, curvature=2, n_basis=5):
    self.D = D
    self.d = d
    self.curvature = curvature
    self.n_basis = n_basis

    # Represent functions f(x): R^d -> R with fourier basis
    # a*cos(pi*<u,x>) + b*sin(pi*<v,x>) for fixed vector u. Thus, represent
    # functions f(x); R^d -> R^D with D such functions


    # NOTE: we use the uniform distribution instead of the normal
    # distribution here because we want to make a hard bound on the
    # curvature. THere are also empirically less numerical problems
    # when staying in the positive region (i.e. rand instead of 2*rand-1)
    
    self.basis_cos = torch.randn(n_basis, D, d)
    # normalize by global 2-norm
    self.basis_cos = self.basis_cos / torch.norm(self.basis_cos, dim=(1,2), keepdim=True) * curvature * torch.sqrt(torch.tensor(d*D, dtype=torch.float32))

    # cutoff of elements of self.basis_cos greater than curvature_normalized
    self.basis_cos[self.basis_cos > curvature] = curvature
    self.basis_cos[self.basis_cos < -curvature] = -curvature
    # measure curvature as size of basis vectors inside trig funcs
    # basis_cos /= (torch.norm(basis_cos, dim=0) / curvature)
    # do same for sin
    self.basis_sin = torch.randn(n_basis, D, d)
    self.basis_sin = self.basis_sin / torch.norm(self.basis_sin, dim=(1,2), keepdim=True) * curvature * torch.sqrt(torch.tensor(d*D, dtype=torch.float32))

    self.basis_sin[self.basis_sin > curvature] = curvature
    self.basis_sin[self.basis_sin < -curvature] = -curvature
    # basis_sin /= (torch.norm(basis_sin, dim=0) / curvature)
    # coefficients of basis functions, random element on unit sphere
    # of norm D

    # We can use the normal distribution here since these scales will be
    # normalized via normalization of the vector field
    self.coeffs_cos = torch.rand(n_basis, D)
    self.coeffs_sin = torch.rand(n_basis, D)

    
# psi of shape (N, d)
  def embedCoordinates(self, psi):
    
    # evaluate the manifold at the coordinates
    # psi: N x d
    # returns: x: N x D

    # evaluate psi on bases. output is (n_basis, D, N)
    basis_eval_cos = torch.matmul(self.basis_cos, psi.T)
    basis_eval_sin = torch.matmul(self.basis_sin, psi.T)
    # evaluate the basis functions
    cos_eval = self.coeffs_cos.unsqueeze(-1) * torch.cos(torch.pi * basis_eval_cos)
    sin_eval = self.coeffs_sin.unsqueeze(-1) * torch.sin(torch.pi * basis_eval_sin)
    # sum over basis functions. output is of shape (D, N)
    x = torch.sum(cos_eval + sin_eval, dim=0)
    return x.T

  def generateSample(self, N, uniform=True):
    # Generate N samples from the manifold
    # N: scalar
    # returns: x in R^D

    # Generate N random coordinates
    if uniform:
      psi = torch.rand((N, self.d))
    else:
      psi = torch.randn((N, self.d))

    # Evaluate the manifold at the coordinates
    x = self.embedCoordinates(psi)
    return x

  
