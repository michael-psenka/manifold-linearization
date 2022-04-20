import torch
from torchdiffeq import odeint

# Class representing a smooth manifold from which we draw data points
# manifold represented as output of a single smooth function

# PROPERTIES:
# D: embedding dimension of manifold
# d: intrinsic dimension of manifold
# curvature: scalar parameter controlling maximum extrinsic curvature
# n_basis: number of basis functions used to represent manifold
class Manifold:
  def __init__(self, D, d, curvature=1, n_basis=5):
    self.D = D
    self.d = d
    self.curvature = curvature
    self.n_basis = n_basis

    # Represent functions f(x): R^D -> R with fourier basis
    # a*cos(pi*<u,x>) + b*sin(pi*<v,x>) for fixed vector u
    curv_param = torch.sqrt(torch.Tensor([self.curvature]))
    self.basis_cos = torch.randn(D, D, n_basis)
    self.basis_cos /= (torch.norm(self.basis_cos, dim=0) / curv_param)
    self.basis_sin = torch.randn(D, D, n_basis)
    self.basis_sin /= (torch.norm(self.basis_sin, dim=0) / curv_param)
    # coefficients of basis functions, random element on unit sphere
    # of norm D
    self.coeffs_cos = torch.randn(D, n_basis)
    self.coeffs_sin = torch.randn(D, n_basis)
    self.coeffs_cos /= (torch.norm(self.coeffs_cos) / D)
    self.coeffs_sin /= (torch.norm(self.coeffs_sin) / D)

  def vecField(self, t, x):
    # Evaluate the vector field at a set of points x
    # x: R^(D x N)
    # returns: v in R^(D x N)

    # Evaluate the basis functions at x
    # (output is of shape (D, N, n_basis))
    basis_eval_cos = (self.basis_cos.reshape((self.D, self.D, 1, self.n_basis)) \
      * x.reshape((self.D, 1, x.shape[1], 1))).sum(dim=0)
    basis_eval_sin = (self.basis_sin.reshape((self.D, self.D, 1, self.n_basis)) \
      * x.reshape((self.D, 1, x.shape[1], 1))).sum(dim=0)

    # Evaluate the vector field at x
    coeffs_reshape_cos = self.coeffs_cos.reshape((self.D, 1, self.n_basis))
    coeffs_reshape_sin = self.coeffs_sin.reshape((self.D, 1, self.n_basis))

    # final evaluation and sum over basis
    return (coeffs_reshape_cos*torch.cos(torch.pi * basis_eval_cos) \
      + coeffs_reshape_sin*torch.sin(torch.pi * basis_eval_sin)).sum(dim=2)

  def embedCoordinates(self, psi):
    # Evaluate the manifold at set of coordinates psi
    # psi: R^(d x N)
    # returns:=x in R^(D x N)
    N = psi.shape[1]
    x_psi = torch.zeros(self.D, N)
    x_psi[:self.d, :] = psi
    t = torch.tensor([0.0,0.3])
    x = odeint(self.vecField, x_psi, t)
    return x[1,:,:]

  def generateSample(self, N, uniform=False):
    # Generate N samples from the manifold
    # N: scalar
    # returns: x in R^D

    # Generate N random coordinates
    if uniform:
      # uniform btwn [-1,1]
      psi = 2*torch.rand((self.d, N)) - 1
    else:
      psi = torch.randn((self.d, N))

    # Evaluate the manifold at the coordinates
    x = self.embedCoordinates(psi)
    return x

  
