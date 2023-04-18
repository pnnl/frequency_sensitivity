# `frequency_sensitivity`

Code for the experiments in [Regularized linear convolutional networks inherit
frequency sensitivity from image statistics](https://arxiv.org/abs/2210.01257).

## Overview

Core functionality computing model gradients with respect to the Fourier basis
as well as various statistics thereof is in `freq_sens.py`. Code for training
the models used in our experiments is in `zoo.py` and its imports. The remaining
scripts are either dependencies of the previous two or used for analysis (e.g.
the aptly named `generate_all_plots.py`).

We gratefully acknowledge dependence on two submodules, namely
[learning_with_noise](https://github.com/mbaradad/learning_with_noise) and
[pytorch-vgg-cifar10](https://github.com/chengyangfu/pytorch-vgg-cifar10.git).

A `conda` environment is specified in `freq_sens.yaml`. In theory it can be
created using the following command.

``` bash
conda env create --file=freq_sens.yaml 
```

## Hyperparameters and model accuracies

Tables of training hyperparameters and model accuracies
can be found in `./model_details`.

## Contribution statement

This repository was extracted from a larger research codebase to which
@nell-byler and @ebishoff made many contributions. In particular, @nell-byler
wrote the first version of `training.py` and both @godfrey-cw and @ebishoff made
further modifications, and `datasets.py` was a collaborative effort of
@godfrey-cw and @nell-byler. The procedural generation (using the wavelet
marginal model) and unsupervised training of AlexNets using
`learning_with_noise` was implemented by @davisrbr. The remainder of the code
was written by @godfrey-cw, although it should be noted that all authors listed
in the citation below contributed substantially in the form of experiment ideas,
feedback, suggestions and debugging advice.

## Citation

If you find this repository useful, please cite:

```bibtex
@article{frequency_sensitivity,
  doi = {10.48550/ARXIV.2210.01257},
  url = {https://arxiv.org/abs/2210.01257},
  author = {Godfrey, Charles and Bishoff, Elise and Mckay, Myles and Brown, Davis and Jorgenson, Grayson and Kvinge, Henry and Byler, Eleanor},
  keywords = {Machine Learning (cs.LG), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Regularized linear convolutional networks inherit frequency sensitivity from image statistics},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

