import sys

# %matplotlib widget
import matplotlib.pyplot as plt
import numpy as np
import torch

from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10

import cc
from models.vae import train_vanilla_vae, train_beta_vae, train_factor_vae
from tools.manifold_generator import Manifold

# magic argparser library thank you github.com/brentyi
import dataclasses
import dcargs

##### COMMAND LINE ARGUMENTS #####
# run cc_test.py --help to get help text
# for each argument below, underscores become dashes
# for example, cc.py --n_iters 1000 is invalid, but
# cc.py --n-iters 1000 is valid

@dataclasses.dataclass
class Args:
	"""cc_test.py
	A python script for testing out the Curvature Compression algorithm."""

	dataset: str = "sine-wave"
	"""A string indicating which dataset to use.
	 Run "python cc_test.py --get-datasets" for a list of possible datasets."""

	get_datasets: bool = False  # If true, print out a list of available datasets and exit.

	get_models: bool = False  # If true, print out a list of available models and exit.

	model: str = "cc"
	"""A string indicating which dataset to use.
	 Run "python cc_test.py --get-datasets" for a list of possible datasets."""

	N: int = 50
	""" Number of training datapoints to use. Note this will not effect some
	 datasets, such as MNIST """

	n_iters: int = 1000 # Number of iterations to run the CC algorithm for

	d_target: int = 1 # dimension we want to flatten the data to

	D: int = 2 # the embedding dimension of the manifold (for random-manifold generation)

	d: int = 1 # the intrinsic dimension of the manifold (for random-manifold generation)

	gamma_0: float = 1
	""" starting value of the "inverse neighborhood size", in the sense that:
	# smaller values of gamma_0 correspond to larger neighborhood sizes, and vice versa """

# NOTE: if adding a new dataset, add it to the list below as well as the big if statement
# inside the main function
datasets = {
  "sine-wave": "The graph of a single period sine wave embedded in 2D",
  "semicircle": "A semicircle of 1.5pi radians embedded in 2D",
  "MNIST": """A single class of the MNIST dataset (in our case, the 2's). This is in spirit to the
		\"union of manifolds\" hypothesis.""",
  "random-manifold": "A random manifold of intrinsic dimension d embedded in D dimensions",
  "CIFAR10": "A single class of the CIFAR10 dataset (the dog class)."
}

models = {
	"cc": "our method, explicitly constructing an encoder/decoder pair using the geometry of the data",
	"vae": "Variational Autoencoders",
	"betavae": "beta-Variational Autoencoders with beta = 4",
	"factorvae": "Factorizing Variational Autoencoders with gamma = 30",
}

if __name__ == "__main__":

	# parse command line arguments
	args = dcargs.cli(Args)

	if args.get_datasets:
		print("Available datasets:\n")
		for dataset, desc in datasets.items():  # dct.iteritems() in Python 2
			print(f"{dataset}: {desc} \n")
		exit(0)
	if args.get_models:
		print("Available models:\n")
		for model, desc in models.items():  # dct.iteritems() in Python 2
			print(f"{dataset}: {desc} \n")
		exit(0)

	# Load the proper dataset
	if args.dataset == "sine-wave":
		N = args.N
		D = 2
		X = torch.zeros((N,D))

		for i in range(N):
			x_coord = i / N*2*torch.pi
			X[i,0] = x_coord
			X[i,1] = np.sin(x_coord)

		# X = X + 0.1*torch.randn(X.shape)

	elif args.dataset == "semicircle":
		N = args.N
		D = 2
		X = torch.zeros((N,D))

		for i in range(N):
			theta_coord = i / N*1.5*torch.pi
			X[i,0] = np.cos(theta_coord)
			X[i,1] = np.sin(theta_coord)

	elif args.dataset == "MNIST":
		dataset = MNIST(root='./torch-dataset', train=True,
								download=True)

		# load dataset into pytorch
		data_loader = torch.utils.data.DataLoader(dataset, batch_size=600000)
		data,labels = next(iter(data_loader))
		data = data.cuda()

		# select single class of dataset
		X = data[labels==2]
		X = X.reshape((5958,32**2))
		X = X.T
	elif args.dataset == "CIFAR10":
		dataset = CIFAR10(root='./torch-dataset', train=True,
								download=True)
		data_loader = torch.utils.data.DataLoader(dataset, batch_size=50000)
		data,labels = next(iter(data_loader))
		data = data.cuda()

		# select single class of dataset
		X = data[labels=='dog']
		X = X.reshape((6000, 32**2))
		X = X.T
	elif args.dataset == "random-manifold":
		N = args.N
		D = args.D
		manifold = Manifold(D, args.d)
		# X = manifold.generateSample(N)
		# coord = torch.zeros((N,args.d))
		# coord[:,0] = torch.linspace(-1,1,N)
		# coord[:,1] = (1 - 1e-4) + torch.linspace(-1e-4,1e-4,N)
		# X = manifold.embedCoordinates(coord)
		X = manifold.generateSample(N)
		# show manifold
		# fig = plt.figure()
		# ax = fig.add_subplot(111, projection='3d')
		# # ax.scatter(X[:int(N / 2), 0], X[:int(N / 2),1], X[:int(N / 2),2], c='b')
		# # ax.scatter(X[int(N / 2):, 0], X[int(N / 2):,1], X[int(N / 2):,2], c='r')
		# ax.scatter(X[:, 0], X[:,1], X[:,2], c='c')
	else:
		sys.exit('Invalid dataset. Run "python cc_test.py --get-datasets" for a list of possible datasets.')


	# center and scale data. not explicitly needed for cc, but helps numerically
	X_mean = torch.mean(X, dim=0)
	X = X - X_mean
	X_norm = torch.norm(X)/np.sqrt(N*D) #expeded norm of gaussian is sqrt(D)
	X = X / X_norm

	if args.model == 'cc':
		f, g = cc.cc(X)
	elif args.model == "vae":
		f, g = train_vanilla_vae(X)
	elif args.model == "betavae":
		f, g = train_beta_vae(X)
	elif args.model == "factorvae":
		f, g = train_factor_vae(X)
	else:
		sys.exit('Invalid model. Run "python cc_test.py --get-models" for a list of possible models.')

	# save features and reconstructions
	Z = f(X).detach()
	# if you want to check interpolation, uncomment the following lines
	# Z_0 = Z[0,:]
	# Z_1 = Z[-1,:]
	# Z_new = torch.zeros((200, Z.shape[1]))
	# for i in range(200):
	# 	Z_new[i,:] = Z_0 + i/200*(Z_1-Z_0)
	X_hat = g(Z).detach()
	# X_hat = g(Z_new)
	# plot the results if possible
	if D == 2:
		plt.scatter(X[:, 0], X[:,1], c='b')
		plt.scatter(Z[:, 0], Z[:,1], c='r')
		plt.scatter(X_hat[:, 0], X_hat[:,1], c='g')
		plt.legend(['X', 'Z', 'Xhat'])
		plt.title(f'Linearization performance of {args.model} on {args.dataset}')
		plt.show()

	elif D == 3:
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(X[:, 0], X[:,1], X[:,2], c='b')
		ax.scatter(Z[:, 0], Z[:,1], Z[:,2], c='r')
		ax.scatter(X_hat[:, 0], X_hat[:,1], X_hat[:,2], c='c')
		ax.legend(['X', 'Z', 'Xhat'])
		ax.set_title(f'Linearization performance of {args.model} on {args.dataset}')
		plt.show()