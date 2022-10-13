import torch
from torchdiffeq import odeint

# Class representing a smooth manifold from which we draw data points
# manifold represented as output of a single smooth function

# HOW THE MANIFOLD IS REPRESENTED: d vector fields over S(D-1) (normalized
# vectors in R^D), denoted here V_i(x) for i = 1,...,d.

# HOW WE SAMPLE FROM THE MANIFOLD: From coordinates psi_1, ..., psi_d:

# 1. with initial condition x_0 = 0, integrate V_1(x) for time psi_1 (possibly
# negative). Denote endint point X_1.
# 2. With initial condition x_0 = X_1, integrate V_2(x) for time psi_2 (possibly
# negative). Denote endint point X_2.
# 3. Repeat above until we have X_d, which is our sample from the manifold.

# PROPERTIES:
# D: embedding dimension of manifold
# d: intrinsic dimension of manifold
# curvature: scalar parameter controlling maximum extrinsic curvature
# n_basis: number of basis functions used to represent manifold
class Manifold:
  def __init__(self, D, d, curvature=0.5, n_basis=3):
    self.D = D
    self.d = d
    self.curvature = curvature
    self.n_basis = n_basis

    # Represent functions f(x): R^D -> R with fourier basis
    # a*cos(pi*<u,x>) + b*sin(pi*<v,x>) for fixed vector u. Thus, represent
    # functions f(x); R^D -> R^D with D such functions

    # Finally we need d such functions, one for each intrinsic dimension
    # we need to move to CPU anyways to integerate separate vec fields, so
    # only helps readability to not mash everything into a single tensor

    # for i in range(d):
    # each coordinate's role:
    # coord 0 cooresponds to the index of the output of the vector field
    # coord 1 corresponds to the index of the inner vector (i.e. u_i)
    # coord 2 corresponds to which basis function we are using

    # normalize the curvauture term by the intrinsic dimension
    curv_normalized = curvature
    # NOTE: we use the uniform distribution instead of the normal
    # distribution here because we want to make a hard bound on the
    # curvature. THere are also empirically less numerical problems
    # when staying in the positive region (i.e. rand instead of 2*rand-1)
    
    self.basis_cos = torch.rand(D, D, n_basis, d)*curv_normalized
    # measure curvature as size of basis vectors inside trig funcs
    # basis_cos /= (torch.norm(basis_cos, dim=0) / curvature)
    # do same for sin
    self.basis_sin = torch.rand(D, D, n_basis, d)*curv_normalized
    # basis_sin /= (torch.norm(basis_sin, dim=0) / curvature)
    # coefficients of basis functions, random element on unit sphere
    # of norm D

    # We can use the normal distribution here since these scales will be
    # normalized via normalization of the vector field
    self.coeffs_cos = torch.rand(D, n_basis, d)
    self.coeffs_sin = torch.rand(D, n_basis, d)

    self.cos_offset = 0.5*torch.rand(1, D, n_basis, d)

  def vecField(self, X, ortho=True):
    # Evaluate the vector field at a set of points X
    # X: R^(N x D)
    # returns: V_dim in R^(N x D x d)

    # Evaluate the basis functions at X
    # (output is of shape (D, N, n_basis))

    # vector field normalized to reduce distortion from coordinate space to
    # embedding space

    # If given the option, use gram-schmidt to orthogonalize the basis vectors
    # at every point 1st basis element

    # only given option since this is quite expensive for higher intrinsic
    # dimensions
    N = X.shape[0]

    # reserve memory for full basis up to (dim)
    # V_full = torch.zeros((N, self.D, dim+1))
    # bases evals, i.e. the (u, v) inside the cos and sin. Output tensor is of shape
    # (N, D, n_basis, d)
    basis_eval_cos = (self.basis_cos.reshape((1, self.D, self.D, self.n_basis, self.d)) \
      * X.reshape((N, 1, self.D, 1, 1))).sum(dim=2)
    # basis_eval_sin = (self.basis_sin.reshape((1, self.D, self.D, self.n_basis, self.d)) \
    #   * X.reshape((N, 1, self.D, 1, 1))).sum(dim=2)

    coeffs_reshape_cos = self.coeffs_cos.reshape((1, self.D, self.n_basis, self.d))
    # coeffs_reshape_sin = self.coeffs_sin.reshape((1, self.D, self.n_basis, self.d))

    # final evaluation and sum over basis. output tensor is of shape (N, D, d)
    V = (coeffs_reshape_cos*torch.cos(self.cos_offset + torch.pi * basis_eval_cos)).sum(dim=2)
      # + coeffs_reshape_sin*torch.sin(1 + torch.pi * basis_eval_sin)).sum(dim=2)
    
    if ortho:
      # gram-schmidt orthogonalization
      V = torch.linalg.qr(V, mode='reduced')[0]

    return V

    
# psi of shape (N, d)
  def embedCoordinates(self, psi):
    # Evaluate the manifold at set of coordinates psi, describes at top of file
    N = psi.shape[0]
    X_t = torch.zeros((N, self.D))

    fid = 10
    # numerical integration along vec field:
    for _ in range(fid):
      # evaluate all vector field at current point
      V_full = self.vecField(X_t, self.d)
      # return desired trajectories from samples. output is of shape (N, D)
      V = (1/fid)* (V_full * psi.reshape((N, 1, self.d))).sum(dim=2)
      # integrate along vector field
      X_t += V
    return X_t

  def generateSample(self, N, uniform=True):
    # Generate N samples from the manifold
    # N: scalar
    # returns: x in R^D

    # Generate N random coordinates
    if uniform:
      # uniform btwn [-1,1]
      psi = torch.rand((N, self.d))
    else:
      psi = torch.randn((N, self.d))

    # Evaluate the manifold at the coordinates
    x = self.embedCoordinates(psi)
    return x

  
