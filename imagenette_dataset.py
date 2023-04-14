"""
modified from
https://docs.neuralmagic.com/sparseml/_modules/sparseml/pytorch/datasets/classification/imagenette.html

"""

import random
import pathlib
from torchvision import transforms
from torchvision.datasets import ImageFolder

__all__ = ["ImagenetteDataset"]


class ImagenetteDataset(ImageFolder):
    """
    Wrapper for the imagenette (10 class) dataset that fastai created.
    Handles downloading and applying standard transforms.

    :param root: The root folder to find the dataset at,
        if not found will download here if download=True
    :param train: True if this is for the training distribution,
        False for the validation
    :param rand_trans: True to apply RandomCrop and RandomHorizontalFlip to the data,
        False otherwise
    :param dataset_size: The size of the dataset to use and download:
        See ImagenetteSize for options
    :param image_size: The image size to output from the dataset
    :param download: True to download the dataset, False otherwise
    """

    def __init__(
        self,
        root: str = "./data/imagenette2",
        split: str = 'train',
        transform = None,
        image_size: int = 224,
    ):
        
        this_dir = pathlib.Path(root) / split

        ImageFolder.__init__(self, str(this_dir), transform)

        self.image_size = image_size

        # make sure we don't preserve the folder structure class order
        random.shuffle(self.samples)