import numpy as np 
import torch as ch
from torch import nn, fft
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision.datasets import CIFAR10, ImageNet, ImageFolder
from torchvision import transforms
import pathlib as pa
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import (ToTensor, ToDevice, ToTorchImage, 
    NormalizeImage, RandomHorizontalFlip, RandomTranslate, Convert)
from ffcv.transforms.common import Squeeze
from ffcv.fields.decoders import (IntDecoder, SimpleRGBImageDecoder, 
    CenterCropRGBImageDecoder, RandomResizedCropRGBImageDecoder)
from imagenette_dataset import ImagenetteDataset

from constants import data_dir

CIFAR_MEAN = [125.307, 122.961, 113.8575]
CIFAR_STD = [51.5865, 50.847, 51.255]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
LWN_MEAN = [0.44087801806139126, 0.42790631331699347, 0.3867879370752931]
LWN_STD = [0.26826768628079806, 0.2610450402318512, 0.26866836876860795]

# requires running their dataset generation code
lwn_dir = 'learning_with_noise/generated_datasets/small_scale/stat-spectrum_color_wmm'

def dataloader_stats(dl: Loader):
    mu: ch.Tensor = None
    sigma: ch.Tensor = None
    for i, (x, _) in enumerate(dl):
        if isinstance(x, list) or isinstance(x, tuple):
            x = ch.cat(x, dim=0)
        if i == 0:
            mu = ch.mean(x, dim=[i for i in range(len(x.shape)) if i != 1])
            sigma = ch.var(x, dim=[i for i in range(len(x.shape)) if i != 1])
        else: 
            mu += ch.mean(x, dim=[i for i in range(len(x.shape)) if i != 1])
            sigma += ch.var(x, dim=[i for i in range(len(x.shape)) if i != 1])

    mu /= len(dl)
    sigma /= len(dl)
    sigma = sigma.sqrt()
    # want to get back a list for compatibility
    return [[x[i] for i in range(x.numel())] for x in [mu, sigma]]

class PassFilter(nn.Module):
    def __init__(self, mode: str = 'high', ratio:float = 1.0) -> None:
        super().__init__()
        self.mode, self.ratio = mode, ratio

    def forward(self, x: ch.Tensor):
        # fft needs at least float32
        old_dtype = x.dtype
        x = x.to(ch.float32)
        xfft = fft.fft2(x, norm='ortho')
        # need to fftshift to be sure we're pass filtering as expected 
        xfft = fft.fftshift(xfft, dim=(-2,-1))
        cutoff: int = None
        if self.mode == 'high':
            originy, originx = int(xfft.shape[-2]/2), int(xfft.shape[-1]/2)
            cutoff = int(self.ratio*xfft.shape[-1]/2)
            # excessive caution
            cutoff = min(cutoff, xfft.shape[-1])
            xfft[..., originy-cutoff:originy+cutoff, originx-cutoff:originx+cutoff] \
                =  ch.zeros_like(xfft[..., originy-cutoff:originy+cutoff, originx-cutoff:originx+cutoff])
            xfft = fft.ifftshift(xfft, dim=(-2,-1))
        # DON'T USE THESE, CURRENTLY BROKEN!!!
        elif self.mode == 'low':
            cutoff = int((1.0-self.ratio)*xfft.shape[-1])
            cutoff = max(cutoff, 0)
            xfft[..., cutoff:, cutoff:] =  ch.zeros_like(xfft[..., cutoff:, cutoff:])
        elif self.mode == 'band':
            cutoff = int(self.ratio*xfft.shape[-1])//2
            cutoff = min(cutoff, xfft.shape[-1]//2)
            xfft[..., xfft.shape[-1]//2-cutoff:xfft.shape[-1]//2+cutoff, xfft.shape[-1]//2-cutoff:xfft.shape[-1]//2+cutoff] =  ch.zeros_like(xfft[..., xfft.shape[-1]//2-cutoff:xfft.shape[-1]//2+cutoff, xfft.shape[-1]//2-cutoff:xfft.shape[-1]//2+cutoff])
        # make sure we stay real ...
        x = fft.ifft2(xfft, norm='ortho').real
        x = x.to(old_dtype)
        return x

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

def subsampledds(ds: dict[str, Dataset] or Dataset, subsample: int = None):
    if isinstance(ds, dict):
        return {k: subsampledds(v, subsample=subsample) for k, v in ds.items()}
    else:
        if subsample == None:
            return ds
        else:
            if len(ds) < subsample:
                print(f'Warning: subsample {subsample} larger than dataset size {len(ds)}')
            idxs = np.random.permutation(len(ds))[:subsample] 
            return Subset(ds, indices=idxs) 

def cifards(subsample: int = None):
    rt = data_dir + '/cifar/'
    rt = pa.Path(rt)
    ds = {
        k: CIFAR10(root=rt, train=(k=='train'), 
            download=True, transform=transforms.ToTensor())
        for k in ['train', 'test']
    }
    if subsample != None:
        ds = subsampledds(ds, subsample=subsample)
    return ds

def cifardl(batch_size:int=512, num_workers:int=4, 
    write=False, half:bool=True,
    subsample: int = None, high_pass: float = None):
    rt = data_dir + '/cifar/'
    rt = pa.Path(rt)
    beton_files = {
        k: (rt / f'cifar_{k}.beton' if not subsample else rt / f'cifar_{k}_{subsample}.beton')
        for k in ['train', 'test']
    }
    # always write when subsampling or if the betons don't exist!
    if any([not v.exists() for v in beton_files.values()]):
        write = True
    if write:
        ds = {
            k: CIFAR10(root=rt, train=(k=='train'), download=True)
            for k in ['train', 'test']
        }
        if subsample != None:
            ds = subsampledds(ds, subsample=subsample)
        writer = {k: DatasetWriter(v,
            {
                'image': RGBImageField(max_resolution=32),
                'label': IntField()
            },
            # num_workers=4
        ) for k, v in beton_files.items()}
        for k in writer.keys():
            writer[k].from_indexed_dataset(ds[k])

    dec = SimpleRGBImageDecoder()
    # NormalizeImage hack to get to [0,1] scale 
    im_pipe = {
        'train': [dec, RandomHorizontalFlip(), RandomTranslate(padding=2,fill=CIFAR_MEAN), 
            ToTensor(), ToDevice(0, non_blocking=True), ToTorchImage(), 
            NormalizeImage(mean=np.zeros(3), std=np.ones(3)*255, type=(np.dtype('float16') if half else np.dtype('float32')))] + ([Convert(ch.float16)] if half else []),
        'test': [dec, ToTensor(), ToDevice(0, non_blocking=True), ToTorchImage(), 
            NormalizeImage(mean=np.zeros(3)*1.0, std=np.ones(3)*255.0, type=(np.dtype('float16') if half else np.dtype('float32')))] + ([Convert(ch.float16)] if half else [])
    }
    if high_pass:
        hpf = PassFilter(mode = 'high', ratio=high_pass)
        im_pipe = {
            k: v + [hpf] for k, v in im_pipe.items()
        }
    lab_pipe = {
        k: [IntDecoder(), ToTensor(), ToDevice(0), Squeeze()]
        for k  in ['train', 'test']
    }
    loaders = {
        k: Loader(
            beton_files[k],
            batch_size=batch_size,
            num_workers=num_workers,
            order=(OrderOption.RANDOM if (k=='train') else OrderOption.SEQUENTIAL),
            drop_last=(k=='train'),
            os_cache=True,
            pipelines={
                'image': im_pipe[k],
                'label': lab_pipe[k]
            },
        ) for k in ['train', 'test']
    }
    return loaders

def imagenetds(subsample: int = None):
    rt = data_dir + '/IN'
    rt = pa.Path(rt)
    pipes = {
        'train': transforms.Compose(
            [   
                transforms.RandomResizedCrop(224, scale=(0.25,1),ratio=(3/4,4/3)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        ),
        'test': transforms.Compose(
            [   
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )
    }
    ds = {
        k: ImageNet(root=rt, split=('train' if k=='train' else 'val'), transform=pipes[k])
        for k in ['train', 'test']
    }
    if subsample != None:
        ds = subsampledds(ds, subsample=subsample)
    return ds

def imagenetdl(batch_size:int = 32, num_workers:int=1,
    subsample: int = None, high_pass: float = None,
    distributed:bool=False, 
    device:int=0, write=False, half:bool=True):
    rt = data_dir + '/IN'
    rt = pa.Path(rt)
    beton_files = {
        k: (rt / f'imagenet_{k}.beton' if not subsample else rt / f'imagenet_{k}_{subsample}.beton') 
        for k in ['train', 'test']
    }
    if any([not v.exists() for v in beton_files.values()]):
        write = True
    if write:
        ds = {
            k: ImageNet(root=rt, split=('train' if k=='train' else 'val'))
            for k in ['train', 'test']
        }
        if subsample != None:
            ds = subsampledds(ds, subsample=subsample)
        writer = {k: DatasetWriter(
            beton_files[k],
            {
                'image': RGBImageField(max_resolution=500),
                'label': IntField()
            },
            # num_workers=4
        ) for k in ['train', 'test']}
        for k in writer.keys():
            writer[k].from_indexed_dataset(ds[k])

    train_dec = RandomResizedCropRGBImageDecoder((224,224), scale=(0.25,1), ratio=(3/4,4/3))
    test_dec = CenterCropRGBImageDecoder((224, 224), ratio=224/256)
    # NormalizeImage hack to get to [0,1] scale 
    im_pipe = {
        'train': [train_dec, RandomHorizontalFlip(), 
            ToTensor(), ToDevice(device, non_blocking=True), ToTorchImage(), 
            NormalizeImage(mean=np.zeros(3), std=np.ones(3)*255, type=(np.dtype('float16') if half else np.dtype('float32')))] + ([Convert(ch.float16)] if half else []),
        'test': [test_dec, ToTensor(), ToDevice(device, non_blocking=True), ToTorchImage(), 
            NormalizeImage(mean=np.zeros(3)*1.0, std=np.ones(3)*255.0, type=(np.dtype('float16') if half else np.dtype('float32')))] + ([Convert(ch.float16)] if half else [])
    }
    if high_pass:
        hpf = PassFilter(mode = 'high', ratio=high_pass)
        im_pipe = {
            k: v + [hpf] for k, v in im_pipe.items()
        }
    lab_pipe = {
        k: [IntDecoder(), ToTensor(), ToDevice(device, non_blocking=True), Squeeze()]
        for k  in ['train', 'test']
    }
    loaders = {
        k: Loader(
            beton_files[k],
            batch_size=batch_size,
            num_workers=num_workers,
            order=OrderOption.RANDOM,
            drop_last=(k=='train'),
            distributed=distributed, 
            os_cache=True,
            # batches_ahead=2,
            pipelines={
                'image': im_pipe[k],
                'label': lab_pipe[k]
            },
        ) for k in ['train', 'test']
    }
    return loaders


def imagenetteds(subsample: int = None):
    rt = data_dir + '/imagenette2'
    rt = pa.Path(rt)
    pipes = {
        'train': transforms.Compose(
            [   
                transforms.RandomResizedCrop(224, scale=(0.25,1),ratio=(3/4,4/3)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        ),
        'test': transforms.Compose(
            [   
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                #transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    }
    ds = {
        k: ImagenetteDataset(root=rt, split=('train' if k=='train' else 'val'), transform=pipes[k])
        for k in ['train', 'test']
    }
    if subsample != None:
        ds = subsampledds(ds, subsample=subsample)
    return ds

def imagenettedl(batch_size:int = 32, num_workers:int=1,
    subsample: int = None, high_pass: float = None,
    distributed:bool=False, 
    device:int=0, write=False, half:bool=True):
    rt = data_dir + '/imagenette2'
    rt = pa.Path(rt)
    # annoying qfs chmod issues ...
    betrt = data_dir + '/imagenette3'
    betrt = pa.Path(betrt)
    beton_files = {
        k: (betrt / f'imagenette_{k}.beton' if not subsample else betrt / f'imagenette_{k}_{subsample}.beton') 
        for k in ['train', 'test']
    }
    if any([not v.exists() for v in beton_files.values()]):
        write = True
    if write:
        ds = {
            k: ImageFolder(root=rt / ('train' if k=='train' else 'val'))
            for k in ['train', 'test']
        }
        if subsample != None:
            ds = subsampledds(ds, subsample=subsample)
        writer = {k: DatasetWriter(
            beton_files[k],
            {
                'image': RGBImageField(max_resolution=500),
                'label': IntField()
            },
            # num_workers=4
        ) for k in ['train', 'test']}
        for k in writer.keys():
            writer[k].from_indexed_dataset(ds[k])

    train_dec = RandomResizedCropRGBImageDecoder((224,224), scale=(0.25,1), ratio=(3/4,4/3))
    test_dec = CenterCropRGBImageDecoder((224, 224), ratio=224/256)
    # NormalizeImage hack to get to [0,1] scale 
    im_pipe = {
        'train': [train_dec, RandomHorizontalFlip(), 
            ToTensor(), ToDevice(device, non_blocking=True), ToTorchImage(), 
            NormalizeImage(mean=np.zeros(3), std=np.ones(3)*255, type=(np.dtype('float16') if half else np.dtype('float32')))] + ([Convert(ch.float16)] if half else []),
        'test': [test_dec, ToTensor(), ToDevice(device, non_blocking=True), ToTorchImage(), 
            NormalizeImage(mean=np.zeros(3)*1.0, std=np.ones(3)*255.0, type=(np.dtype('float16') if half else np.dtype('float32')))] + ([Convert(ch.float16)] if half else [])
    }
    if high_pass:
        hpf = PassFilter(mode = 'high', ratio=high_pass)
        im_pipe = {
            k: v + [hpf] for k, v in im_pipe.items()
        }
    lab_pipe = {
        k: [IntDecoder(), ToTensor(), ToDevice(device, non_blocking=True), Squeeze()]
        for k  in ['train', 'test']
    }
    loaders = {
        k: Loader(
            beton_files[k],
            batch_size=batch_size,
            num_workers=num_workers,
            order=OrderOption.RANDOM,
            drop_last=(k=='train'),
            distributed=distributed, 
            os_cache=True,
            # batches_ahead=2,
            pipelines={
                'image': im_pipe[k],
                'label': lab_pipe[k]
            },
        ) for k in ['train', 'test']
    }
    return loaders


def lwnds(alpha:float=1.0):
    transform_array = [
        transforms.RandomResizedCrop(64, scale=(0.08, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
    ]
    t = transforms.Compose(transform_array)
    t = TwoCropsTransform(t)
    rt = lwn_dir / f'train_{str(alpha)}'
    print(rt)
    ds = {
        k: ImageFolder(root=rt, transform=t)
        for k in ['train', 'test']
    }
    return ds

def lwndl(alpha:float=1.0, subsample:float=None,
    batch_size:int = 32, num_workers:int=1, half:bool=False):
    ds = lwnds(alpha=alpha)
    if subsample != None:
        ds = subsampledds(ds=ds, subsample=subsample)
    dl = {
        k: DataLoader(v, batch_size=batch_size, num_workers=num_workers,
            shuffle=True, drop_last=False)
        for k, v in ds.items()
    }
    return dl