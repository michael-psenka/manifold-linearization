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
  def __init__(self, D, d, curvature=0.6, n_basis=3):
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
    self.basis_cos = []
    self.basis_sin = []
    self.coeffs_cos = []
    self.coeffs_sin = []

    for i in range(d):
      # each coordinate's role:
      # coord 0 cooresponds to the index of the output of the vector field
      # coord 1 corresponds to the index of the inner vector (i.e. u_i)
      # coord 2 corresponds to which basis function we are using
      basis_cos = torch.randn(D, D, n_basis)
      # measure curvature as size of basis vectors inside trig funcs
      basis_cos /= (torch.norm(basis_cos, dim=0) / curvature)
      self.basis_cos.append(basis_cos)
      # do same for sin
      basis_sin = torch.randn(D, D, n_basis)
      basis_sin /= (torch.norm(basis_sin, dim=0) / curvature)
      self.basis_sin.append(basis_sin)
      # coefficients of basis functions, random element on unit sphere
      # of norm D
      coeffs_cos = torch.randn(D, n_basis)
      coeffs_sin = torch.randn(D, n_basis)

      self.coeffs_cos.append(coeffs_cos)
      self.coeffs_sin.append(coeffs_sin)

  def vecField(self, X, dim, ortho=True):
    # Evaluate the vector field at a set of points X
    # X: R^(N x D)
    # returns: V_dim in R^(N x D)

    # Evaluate the basis functions at X
    # (output is of shape (D, N, n_basis))

    # vector field normalized to reduce distortion from coordinate space to
    # embedding space

    # If given the option, use gram-schmidt to orthogonalize the basis vectors
    # at every point 1st basis element

    # only given option since this is quite expensive for higher intrinsic
    # dimensions
    N = X.shape[0]

    if ortho:
      # reserve memory for full basis up to (dim)
      V_full = torch.zeros((N, self.D, dim+1))
      # bases evals, i.e. the (u, v) inside the cos and sin. Output tensor is of shape
      # (N, D, n_basis)
      basis_eval_cos = (self.basis_cos[0].reshape((1, self.D, self.D, self.n_basis)) \
        * X.reshape((N, 1, self.D, 1))).sum(dim=2)
      basis_eval_sin = (self.basis_sin[0].reshape((1, self.D, self.D, self.n_basis)) \
        * X.reshape((N, 1, self.D, 1))).sum(dim=2)

      coeffs_reshape_cos = self.coeffs_cos[0].reshape((1, self.D, self.n_basis))
      coeffs_reshape_sin = self.coeffs_sin[0].reshape((1, self.D, self.n_basis))

      # final evaluation and sum over basis. output tensor is of shape (N, D)
      V = (coeffs_reshape_cos*torch.cos(torch.pi * basis_eval_cos) \
        + coeffs_reshape_sin*torch.sin(torch.pi * basis_eval_sin)).sum(dim=2)

      V = V / V.norm(dim=1, keepdim=True)
      V_full[:, :, 0] = V
      
      # apply gram-schmidt until we get to our desired dimension
      for i in range(1, dim+1):
        # get the new vector field
        basis_eval_cos = (self.basis_cos[i].reshape((1, self.D, self.D, self.n_basis)) \
          * X.reshape((N, 1, self.D, 1))).sum(dim=2)
        basis_eval_sin = (self.basis_sin[i].reshape((1, self.D, self.D, self.n_basis)) \
          * X.reshape((N, 1, self.D, 1))).sum(dim=2)

        coeffs_reshape_cos = self.coeffs_cos[i].reshape((1, self.D, self.n_basis))
        coeffs_reshape_sin = self.coeffs_sin[i].reshape((1, self.D, self.n_basis))

        # final evaluation and sum over basis. output tensor is of shape (N, D)
        V = (coeffs_reshape_cos*torch.cos(torch.pi * basis_eval_cos) \
          + coeffs_reshape_sin*torch.sin(torch.pi * basis_eval_sin)).sum(dim=2)

        # orthogonalize, V_i=  V_i - V_<i V_<i^T V_i for all i \in [N]
        # TODO: check correctness of this
        V = V - \
          ((V.reshape((N, self.D, 1)) * V_full[:,:,:i]).sum(dim=1, keepdim=True) * V_full[:,:,:i]).sum(dim=2)
        V = V / V.norm(dim=1, keepdim=True)

        # store for future gram schmidt iterations
        V_full[:, :, i] = V

      # normalize tanget vectors to control norm distortion when integrating
      return V_full[:, :, -1]

    else:
      basis_eval_cos = (self.basis_cos[dim].reshape((1, self.D, self.D, self.n_basis)) \
        * X.reshape((N, 1, self.D, 1))).sum(dim=2)
      basis_eval_sin = (self.basis_sin[dim].reshape((1, self.D, self.D, self.n_basis)) \
        * X.reshape((N, 1, self.D, 1))).sum(dim=2)

      coeffs_reshape_cos = self.coeffs_cos[dim].reshape((1, self.D, self.n_basis))
      coeffs_reshape_sin = self.coeffs_sin[dim].reshape((1, self.D, self.n_basis))

      # final evaluation and sum over basis. output tensor is of shape (N, D)
      V = (coeffs_reshape_cos*torch.cos(torch.pi * basis_eval_cos) \
        + coeffs_reshape_sin*torch.sin(torch.pi * basis_eval_sin)).sum(dim=2)

      return V / V.norm(dim=1, keepdim=True)

  def embedCoordinates(self, psi):
    # Evaluate the manifold at set of coordinates psi, describes at top of file

    # psi: R^(N x d)
    # returns: X in R^(N x D)
    N = psi.shape[0]
    # set up initial condition
    X_t = torch.zeros((N, self.D))

    for i in range(self.d):
      # integrate the vector field for time psi[:, i]
      f = lambda t, X: self.vecField(X, dim=i)
      # will need to handle negative time separately
      f_neg = lambda t, X: -self.vecField(X, dim=i)

      t, t_idx = psi[:, i].sort()

      t_pos = t[t>0]
      # solution is of shape (t.numel() ,N, D)
      sol_pos = odeint(f, X_t, t_pos, method='rk4', rtol=1e-6, atol=1e-6)
      # note that input time needs to be positive and sorted, can recover
      # from full sorted list of times as follows
      t_neg = -t[t<= 0].flip(dims=[0])
      sol_neg = odeint(f_neg, X_t, t_neg, method='rk4', rtol=1e-6, atol=1e-6)
      sol_neg = sol_neg.flip(dims=[0])
      # note above we did a little overkill, evaluate all times for all points,
      # but we only want each point's corresponding time

      # (note diag is weird and outputs the diagonalized index last)
      X_t[t_idx[t>0],:] = sol_pos[:,t_idx[t>0],:].diagonal(dim1=0, dim2=1).T
      X_t[t_idx[t<=0],:] = sol_neg[:,t_idx[t<=0],:].diagonal(dim1=0, dim2=1).T

    return X_t

  def generateSample(self, N, uniform=True):
    # Generate N samples from the manifold
    # N: scalar
    # returns: x in R^D

    # Generate N random coordinates
    if uniform:
      # uniform btwn [-1,1]
      psi = 2*torch.rand((N, self.d)) - 1
    else:
      psi = torch.randn((N, self.d))

    # Evaluate the manifold at the coordinates
    x = self.embedCoordinates(psi)
    return x

  
