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

