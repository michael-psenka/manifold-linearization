# Manifold Linearization for Representation Learning

This is a research project focused on the automatic generation of autoencoders with minimal feature size, when the data is supported near an embedded submanifold. Using the geometric structure of the manifold, we can equivalently treat this problem as a manifold flattening problem when the manifold is flattenable[^1]. See our paper, [Representation Learning via Manifold Flattening and Reconstruction](https://www.michaelpsenka.io/papers/flatnet/), for more details.

The main practical benefit: *neural network autoencoders that build themselves from your data!* Instead of trial-and-error guessing what architecture dimensions to use, simply run a geometric process that automatically optimizes layer width, depth, and stopping time based on your data[^2].

[^1]: Geometric note: while flattenability is not general, there are some heuristic reasons we can motivate this assumption for real world data. For example, if a dataset permits a VAE-like autoencoder, where samples from the data distribution can be generated via a standard Gaussian in the latent space, then the samples lie close in probability to a flattenable manifold, as this VAE has constructed a single-chart atlas.

[^2]: Of course, there is no free lunch. There are a still a few scalar hyperparameters that need to be set, but (a) this search space is drastically lower-dimensional than choosing the entire architecture structure, and (b) these are intuitive to choose and are incredibly robust (for example, these were hardly changed between the experiments on synthetic manifolds and MNIST).

## Installation

The pure FlatNet construction (training) code is available as a pip package and can be installed in the following way:

```
pip install flatnet
```

### Optional installation

This repo also contains a number of testing and illustrative files to both familiarize new users with the framework and show the experiments run in the main paper. To install the appropriate remaining dependencies for this repo, first navigate to the project directory, then run the following command:

```
pip install -r requirements.txt
```

## Quickstart usage

The follolwing is a simple example script using the pip package:

```python
import torch
import flatnet

# create sine wave dataset
t = torch.linspace(0, 1, 50)
y = torch.sin(t * 2 * 3.14)

# format dataset of N points of dimension D as (N, D) matrix
X = torch.stack([t, y], dim=1)
# add noise
X = X + 0.02*torch.randn(*X.shape)

# normalize data
X = (X - X.mean(dim=0)) / X.std(dim=0)

# train encoder f and decoder g, then save a gif of the
# manifold evolution through the constructed layers of f
f, g = flatnet.train(X, n_iter=150, save_gif=True)
```

The script `flatnet_test.py` includes many example experiments to run FlatNet constructions on. To see an example experiment, simply run `python flatnet_test.py` in the main directory to see the flattening and reconstruction of a simple sine wave. Further experiments and options can be specified through command line arguments, managed through [tyro](https://github.com/brentyi/tyro); to see the full list of arguments, run `python flatnet_test.py --help`.


## Directory Structure

- `flatnet_test.py`: main test script, as described in above section.
- `flatnet/train.py`: contains the main FlatNet construction (training) code.
- `flatnet/modules`: contains code for the neural network modules used in FlatNet.
- `experiments-paper`: contains scripts and results from experiments done in the paper.
- `models`: contains code for various models that FlatNet was compared against in the paper.
- `tools`: contains auxillery tools for evaulating the method, such as random manifold generators -- one of which being from the [randman](https://github.com/fzenke/randman) repo.


## Citation

If you use this work in your research, please cite the following paper:

```
@article{psenka2023flatnet,
  author = {Psenka, Michael and Pai, Druv and Raman, Vishal and Sastry, Shankar and Ma, Yi},
  title = {Representation Learning via Manifold Flattening and Reconstruction},
  year = {2023},
  eprint = {2305.01777},
  url = {https://arxiv.org/abs/2305.01777},
}
```

We hope that you find this project useful. If you have any questions or suggestions, please feel free to contact us.

## Copyright notice

THIS SOFTWARE AND/OR DATA WAS DEPOSITED IN THE BAIR OPEN RESEARCH COMMONS REPOSITORY ON 5/3/2023.
