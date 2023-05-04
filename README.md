# Autoencoder Generator

This is a research project focused on the automatic generation of autoencoders with minimal feature size, when the data is supported near an embedded submanifold. Using the geometric structure of the manifold, we can equivalently treat this problem as a manifold flattening problem. See our paper, [Representation Learning via Manifold Flattening and Reconstruction](https://arxiv.org/abs/2305.01777), for more details.

## Installation

The project is written in Python. To install the appropriate dependencies, run the following command:

```
pip install -r requirements.txt
```

## Quickstart usage

The script `cc_test.py` includes many example experiments to run FlatNet constructions on. To see an example experiment, simply run `python cc_test.py` in the main directory to see the flattening and reconstruction of a simple sine wave. Further experiments and options can be specified through command line arguments, managed through [tyro](https://github.com/brentyi/tyro); to see the full list of arguments, run `python cc_test.py --help`.


## Directory Structure


- `experiments-paper`: contains scripts and results from experiments done in the paper.
- `models`: contains code for various models that FlatNet was compared against in the paper.
- `test-files`: contains
- `README.md`: this file.
- `requirements.txt`: contains the list of required dependencies.

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