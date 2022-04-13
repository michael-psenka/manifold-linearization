import torch

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
    # cos(pi*<u,x>) for fixed vector u
    curv_param = torch.sqrt(torch.Tensor([self.curvature]))
    self.basis = curv_param*torch.randn(d, D, n_basis)
    # coefficients of basis functions, random element on unit sphere
    # of norm D
    self.coeffs = torch.randn(D, n_basis)
    self.coeffs /= torch.norm(self.coeffs) / D

  def embedCoordinates(self, psi):
    # TODO: add sin component
    # Evaluate the manifold at set of coordinates psi
    # psi: R^(d x N)
    # returns:=x in R^(D x N)

    N = psi.shape[1]

    # Evaluate the basis functions at psi
    # (output is of shape (D, N, n_basis))
    basis_eval = (self.basis.reshape((self.d, self.D, 1, self.n_basis)) \
      * psi.reshape((self.d, 1, N, 1))).sum(dim=0)
    print(basis_eval.shape)

    # Evaluate the manifold at psi
    coeffs_reshape = self.coeffs.reshape((self.D, 1, self.n_basis))
    print((coeffs_reshape*torch.sin(torch.pi * basis_eval)).sum(dim=2).shape)
    return (coeffs_reshape*torch.sin(torch.pi * basis_eval)).sum(dim=2)

  def generateSample(self, N, uniform=False):
    # Generate N samples from the manifold
    # N: scalar
    # returns: x in R^D

    # Generate N random coordinates
    if uniform:
      psi = torch.rand((self.d, N))
    else:
      psi = torch.randn((self.d, N))

    # Evaluate the manifold at the coordinates
    x = self.embedCoordinates(psi)
    return x

  
