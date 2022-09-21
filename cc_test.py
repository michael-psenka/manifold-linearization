import sys

# %matplotlib widget
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms


import cc

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

	model: str = "cc"
	"""A string indicating which dataset to use.
	 Run "python cc_test.py --get-datasets" for a list of possible datasets."""

	get_models: bool = False  # If true, print out a list of available datasets and exit.

	N: int = 100
	""" Number of training datapoints to use. Note this will not effect some
	 datasets, such as MNIST """

	n_iters: int = 1000 # Number of iterations to run the CC algorithm for

	d_target: int = 1 # dimension we want to flatten the data to

	gamma_0: float = 1
	""" starting value of the "inverse neighborhood size", in the sense that:
	# smaller values of gamma_0 correspond to larger neighborhood sizes, and vice versa """

# NOTE: if adding a new dataset, add it to 1the list below as well as the big if statement
# inside the main function
datasets = {
  "sine wave": "The graph of a single period sine wave embedded in 2D",
  "semicircle": "A semicircle of 1.5pi radians embedded in 2D",
  "MNIST": """A single class of the MNIST dataset (in our case, the 2's). This is in spirit to the
		\"union of manifolds\" hypothesis.""",
}

models = {
	"cc": "our method, explicitly constructing an encoder/decoder pair using the geometry of the data"
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

	elif args.dataset == "semicircle":
		N = args.N
		D = 2
		X = torch.zeros((N,D))

		for i in range(N):
			theta_coord = i / N*1.5*torch.pi
			X[i,0] = np.cos(theta_coord)
			X[i,1] = np.sin(theta_coord)

	elif args.dataset == "MNIST":
		dataset = datasets.MNIST(root='./torch-dataset', train=True,
								download=True)

		# load dataset into pytorch
		data_loader = torch.utils.data.DataLoader(dataset, batch_size=600000)
		data,labels = next(iter(data_loader))
		data = data.cuda()

		# select single class of dataset
		X = data[labels==2]
		X = X.reshape((5958,32**2))
		X = X.T
	else:
		sys.exit('Invalid dataset. Run "python cc_test.py --get-datasets" for a list of possible datasets.')


	# center and scale data. not explicitly needed for cc, but helps numerically
	X_mean = torch.mean(X, dim=0)
	X = X - X_mean
	X_norm = torch.norm(X)/np.sqrt(N*D) #expeded norm of gaussian is sqrt(D)
	X = X / X_norm

	if args.model == 'cc':
		Z = cc.cc(X)


	plt.plot(Z[:,0], Z[:,1], '.')
	plt.show()