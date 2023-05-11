from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("flatnet/CHANGES.rst", "r", encoding="utf-8") as ch:
    changelog = ch.read()

long_desc_init = 'This is a minimal pip package to allow easy deployment of FlatNets, a geometry-based neural autoencoder architecture that automatically builds its layers based on geometric properties of the dataset. See our paper, [Representation Learning via Manifold Flattening and Reconstruction](https://arxiv.org/abs/2305.01777), for more details, and [the Github repo](https://github.com/michael-psenka/manifold-linearization) for the code and example scripts & notebooks.'

long_description = long_desc_init + "\n\n" + changelog + "\n\n" + long_description

setup(
    name='flatnet',
    version='0.2.1',
    description='FlatNet implementation in PyTorch, from the paper \"Representation Learning via Manifold Flattening and Reconstruction\"',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Michael Psenka',
    author_email='psenka@eecs.berkeley.edu',
    maintainer='Druv Pai <druvpai@berkeley.edu>, Vishal Raman <vraman@berkeley.edu>',
    packages=find_packages(),
    install_requires=[ 
        'cmake>=3.26.3',
        'filelock>=3.12.0',
        'imageio>=2.28.1',
        'geoopt>=0.5.0',
        'Jinja2>=3.1.2',
        'lit>=16.0.2',
        'MarkupSafe>=2.1.2',
        'mpmath>=1.3.0',
        'networkx>=3.1',
        'numpy>=1.24.3',
        'nvidia-cublas-cu11>=11.10.3.66',
        'nvidia-cuda-cupti-cu11>=11.7.101',
        'nvidia-cuda-nvrtc-cu11>=11.7.99',
        'nvidia-cuda-runtime-cu11>=11.7.99',
        'nvidia-cudnn-cu11>=8.5.0.96',
        'nvidia-cufft-cu11>=10.9.0.58',
        'nvidia-curand-cu11>=10.2.10.91',
        'nvidia-cusolver-cu11>=11.4.0.1',
        'nvidia-cusparse-cu11>=11.7.4.91',
        'nvidia-nccl-cu11>=2.14.3',
        'nvidia-nvtx-cu11>=11.7.91',
        'scipy>=1.10.1',
        'sympy>=1.11.1',
        'torch>=2.0.0',
        'tqdm>=4.65.0',
        'triton>=2.0.0',
        'typing_extensions>=4.5.0',
    ],
)