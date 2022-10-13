'''
From Zenke, F., and Vogels, T.P. (2021). The Remarkable Robustness of Surrogate Gradient Learning for 
Instilling Complex Function in Spiking Neural Networks. Neural Computation 1–27.
'''

import numpy as np
import torch

class RandMan:
    """ Randman objects hold the parameters for a smooth random manifold from which datapoints can be sampled. """
    
    def __init__(self, embedding_dim, manifold_dim, alpha=2, n_basis=3, prec=1e-3, max_f_cutoff=1000, seed=None, dtype=torch.float32, device=None):
        """ Initializes a randman object.
        
        Args
        ----
        D : extrinsic dimension of manifold
        d : intrinsic dimension of manifold
        alpha : The power spectrum fall-off exponenent(inversely proportional to extrinsic curvature)  Determines the smoothenss of the manifold (default 2)
        seed: This seed is used to init the *global* torch.random random number generator. 
        prec: The precision paramter to determine the maximum frequency cutoff (default 1e-3)
        """
        self.alpha = alpha
        self.D = embedding_dim
        self.d = manifold_dim
        self.f_cutoff = int(np.min((np.ceil(np.power(prec,-1/self.alpha)),max_f_cutoff)))
        self.n_basis = n_basis
        self.dtype=dtype
        
        if device is None:
            self.device=torch.device("cpu")
        else:
            self.device=device
        
        if seed is not None:
            torch.random.manual_seed(seed)

        self.init_random()
        self.init_spect(self.alpha)
           
    def init_random(self):
        self.params = torch.rand(self.D, self.d, self.n_basis, self.f_cutoff, dtype=self.dtype, device=self.device)
        self.params[:,:,0,0] = 0

    def init_spect(self, alpha=2.0, res=0, ):
        """ Sets up power spectrum modulation 
        
        Args
        ----
        alpha : Power law decay exponent of power spectrum
        res : Peak value of power spectrum.
        """
        r = (torch.arange(self.f_cutoff,dtype=self.dtype,device=self.device)+1)
        s = 1.0/(torch.abs(r-res)**alpha + 1.0)
        self.spect = s
        
    def eval_random_function_1d(self, x, theta):       
        tmp = torch.zeros(len(x),dtype=self.dtype,device=self.device)
        s = self.spect
        for i in range(self.f_cutoff):
            tmp += theta[0,i]*s[i]*torch.sin( 2*np.pi*(i*x*theta[1,i] + theta[2,i]) )
        return tmp

    def eval_random_function(self, x, params):
        tmp = torch.ones(len(x),dtype=self.dtype,device=self.device)
        for i in range(self.d):
            tmp *= self.eval_random_function_1d(x[:,i], params[i])
        return tmp
    
    def eval_manifold(self, x):
        if isinstance(x,np.ndarray):
            x = torch.tensor(x,dtype=self.dtype,device=self.device)
        tmp = torch.zeros((x.shape[0],self.D),dtype=self.dtype,device=self.device)
        for i in range(self.D):
            tmp[:,i] = self.eval_random_function(x, self.params[i])
        return tmp
    
    def generateSample(self, N):
        x = torch.rand(N, self.d, dtype=self.dtype,device=self.device)
        y = self.eval_manifold(x)
        return y
