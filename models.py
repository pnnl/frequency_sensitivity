import torch as ch
from torch import nn
from torch.nn.init import kaiming_normal_
import numpy as np
from torchvision import models as tvmods
from composer.models import ComposerResNetCIFAR
from datasets import (CIFAR_MEAN, CIFAR_STD, 
    IMAGENET_MEAN, IMAGENET_STD,
    LWN_MEAN, LWN_STD)
from typing import Callable
from dataclasses import dataclass, field
from pytorch_vgg_cifar10.vgg import (
    vgg11 as vgg_cifar11,
    vgg13 as vgg_cifar13,
    vgg16 as vgg_cifar16,
    vgg19 as vgg_cifar19,
    VGG as VGG_cifar
)
from learning_with_noise.align_uniform.encoder import SmallAlexNet 

from constants import results_dir

vgg_cifar_dict = {
    11: vgg_cifar11,
    13: vgg_cifar13,
    16: vgg_cifar16,
    19: vgg_cifar19
}

class Nrmlz(Callable):
    def __init__(self, mean: list[float], std: list[float],
        div_by_255: bool=False) -> None:
        super().__init__()
        self.mean, self.std = mean, std
        self.div_by_255= div_by_255

    def __call__(self, x: ch.Tensor) -> ch.Tensor:
        m = ch.tensor(self.mean).reshape((3,1,1)).to(x.device)
        s = ch.tensor(self.std).reshape((3,1,1)).to(x.device)
        if self.div_by_255:
            m, s = m/255.0, s/255.0
        return (x-m)/s

cifar_nrmlz = Nrmlz(mean = CIFAR_MEAN, std = CIFAR_STD, div_by_255=True)
inet_nrmlz = Nrmlz(mean= IMAGENET_MEAN, std = IMAGENET_STD, div_by_255=False)

class RNC(nn.Module):
    def __init__(self, depth:int=9, 
        mean: list[float] = CIFAR_MEAN, 
        std: list[float] = CIFAR_STD, 
        div_by_255:bool=True):
        super().__init__()
        self.nrmlz = Nrmlz(mean=mean, std=std, div_by_255=div_by_255)
        assert depth in [9, 20, 56], f'valid depths are 9, 20, 56 but got {depth}'
        self.rn = ComposerResNetCIFAR(f'resnet_{depth}')

    def forward(self, x):
        x = self.nrmlz(x)
        # annoying: composer models take x, y as *input* (??!)
        return self.rn((x, ch.empty((x.shape[0],))))

def rnc(depth:int=9, pretrained:bool=False, 
    mean: list[float] = CIFAR_MEAN, std: list[float] = CIFAR_STD, div_by_255:bool=True):
    model = RNC(depth=depth, mean=mean, std=std, div_by_255=div_by_255)
    if pretrained:
        assert depth==9, 'only have pretrained resnet_9 :('
        bw = ch.load(results_dir/'freq-sens' / 'resnet_cifar'/'best_wts.pt')
        model.load_state_dict(bw)
    model = model.to(memory_format=ch.channels_last)
    return model

class RN(nn.Module):
    def __init__(self, depth:int=50, 
        mean: list[float] = IMAGENET_MEAN, 
        std: list[float] = IMAGENET_STD, 
        div_by_255:bool=False) -> None:
        super().__init__()
        self.nrmlz = Nrmlz(mean=mean, std=std, div_by_255=div_by_255)
        depths = [18, 34, 50, 101, 152]
        assert depth in depths, f'valid depths are {depths} but got {depth}'
        self.rn = getattr(tvmods, f'resnet{depth}')

    def forward(self, x):
        x = self.nrmlz(x)
        return self.rn(x)

def rn(depth=50, pretrained:bool=False,
    mean: list[float] = IMAGENET_MEAN, std: list[float] = IMAGENET_STD, div_by_255:bool=False):
    model = RN(depth=depth, mean=mean, std=std, div_by_255=div_by_255)
    if pretrained:
        assert depth==50, 'only have pretrained resnet_50 :('
        bw = ch.load(results_dir/'freq-sens' / 'resnet_imagenet'/'best_wts.pt')
        # hack to deal with annoying aspects of dist. training
        bw = {'.'.join(k.split('.')[1:]): v for k, v in bw.items()}
        model.load_state_dict(bw)
    model = model.to(memory_format=ch.channels_last)
    return model

# Myrtle stuff
class Flatten(nn.Module):
    def forward(self, x):
        return x.view((x.size(0), -1))
    
class mCNN_k(nn.Module):
    def __init__(self, c=64, num_classes=10,
        mean: list[float] = CIFAR_MEAN, std: list[float] = CIFAR_STD, div_by_255:bool=True) -> None:
        super().__init__()
        self.nrmlz = Nrmlz(mean=mean, std=std, div_by_255=div_by_255)
        self.c1 = nn.Sequential(
            nn.Conv2d(3, c, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.cs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c*(2**i), c*(2**(i+1)), kernel_size=3, stride=1, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )
                for i in range(3)
            ]
        )
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            Flatten(),
            nn.Linear(c * 8, num_classes, bias=True)
        )
        
    def forward(self, x:ch.Tensor):
        x= self.nrmlz(x)
        x = self.c1(x)
        for c in self.cs:
            x = c(x)
        return self.classifier(x)

def mcnnk(pretrained:bool=False, 
    mean: list[float] = CIFAR_MEAN, 
    std: list[float] = CIFAR_STD, div_by_255:bool=True):
    model =  mCNN_k(mean=mean, std=std, div_by_255=div_by_255)
    model = model.to(memory_format=ch.channels_last)
    return model

@dataclass(eq=False)
class ConvActually(nn.Module):
    in_channels: int = 1
    out_channels: int = 1
    kernel_size: int = 1
    pool_size: int = None
    batch_norm: bool = False
    identity_activations: bool = False
    debug: bool = False

    def __post_init__(self):
        super().__init__()
        self.activation: nn.Module = (nn.Identity() if self.identity_activations else nn.ReLU())
        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, 
            kernel_size=self.kernel_size, padding='same', padding_mode='circular', 
            bias=(not self.batch_norm))
        kaiming_normal_(self.conv.weight, nonlinearity='relu')
        self.bn: nn.Module
        self.bn = (nn.BatchNorm2d(self.out_channels) if self.batch_norm else nn.Identity())
        if self.batch_norm:
            kaiming_normal_(self.bn.weight)
        self.pool: nn.Module
        if self.pool_size:
            if self.conv.kernel_size % 2 != 0:
                self.pool = nn.MaxPool2d(kernel_size=self.pool_size, padding=(self.kernel_size - 1)//2)
            else: 
                self.pool = nn.MaxPool2d(kernel_size=self.pool_size, padding=self.kernel_size//2)
    
    def forward(self, x:ch.Tensor):
        x = self.conv(x)
        x = self.activation(self.bn(x))
        if self.debug:
            print(x.shape)
        if self.pool_size:
            x = self.pool(x)
            if self.conv.kernel_size % 2 != 0:
                x = x[..., :self.kernel_size, :self.kernel_size]
            if self.debug:
                print(x.shape)
        return x

@dataclass(eq=False)
class Vectorizer(nn.Module):
    mode: str = 'flatten'

    def __post_init__(self):
        super().__init__()

    def forward(self, x:ch.Tensor):
        if self.mode == 'flatten':
            if len(x.shape) > 3:
                return x.reshape((x.shape[0], -1))
            else:
                return x.reshape((-1,))
        elif self.mode == 'average':
            x = x.reshape(x.shape[:-2] + (-1,))
            return x.mean(dim=-1)
        elif self.mode == 'max':
            x = x.reshape(x.shape[:-2] + (-1,))
            return x.max(dim=-1)[0]
        else:
            raise NotImplementedError(f'mode {self.mode} not implemented')
        
@dataclass(eq=False)
class CNNActually(nn.Module):
    input_resolution: int = 32
    identity_activations: bool=False
    channels: tuple[int] = tuple([2**i for i in range(4, 7)])
    classes: int = 1
    pool_size: int = None
    batch_norm: bool = False
    feature_extractor: str = 'flatten'
    mean: list[float] = field(default_factory=list)
    std: list[float] = field(default_factory=list)
    div_by_255:bool=True
    debug: bool = False

    def __post_init__(self):
        super().__init__()
        self.activation = (nn.Identity() if self.identity_activations else nn.ReLU())
        self.channels = list(self.channels)
        self.channels = [3] + self.channels
        if self.debug:
            print('channels', self.channels)
        self.nrmlz = Nrmlz(mean=self.mean, std=self.std, div_by_255=self.div_by_255)
        self.convblocks = nn.ModuleList([ 
            ConvActually(in_channels=self.channels[i], out_channels=self.channels[i+1], 
                kernel_size=self.input_resolution,
                pool_size=self.pool_size, batch_norm=self.batch_norm,
                identity_activations=self.identity_activations,
                debug = self.debug)
            for i in range(len(self.channels) - 1)
        ])
        self.vec = Vectorizer(mode=self.feature_extractor)
        fxd = self.fx_dim()
        self.lin = nn.Linear(fxd, self.classes, bias=True)
        kaiming_normal_(self.lin.weight, nonlinearity='relu')

    def fx(self, x:ch.Tensor):
        x = self.nrmlz(x=x)
        for i, l in enumerate(self.convblocks):
            if self.debug:
                print(f'layer {i} incoming shape: ', x.shape)
            x = l(x)
            if self.debug:
                print(f'layer {i} outgoing shape: ', x.shape)
        return self.vec(x)
    
    def fx_dim(self):
        x = ch.empty((2, 3, self.input_resolution, self.input_resolution))
        y = self.fx(x)
        return y.shape[-1]
    
    def forward(self, x:ch.Tensor):
        return self.lin(self.fx(x))

def cnn_actually(input_resolution:int = 32, 
    identity_activations: bool = False,
    depth:int=4, layer_one_width:int = 64, 
    layer_width_doublings: int = None, exp_width:bool=False,
    classes=10, feature_extractor='flatten', batch_norm:bool=False,
    debug: bool= False, pretrained: bool=False,
    mean: list[float] = CIFAR_MEAN, std: list[float] = CIFAR_STD, 
    div_by_255:bool=True):
    assert not exp_width or layer_width_doublings, f'if exp_width I need layer_width_doublings.'
    channels: list[int] =  None
    if exp_width:
        channels = [layer_one_width*int(x) 
            for x in 2**np.linspace(0, layer_width_doublings, depth-1)]
    else:
        channels = [layer_one_width for i in range(depth-1)]
    channels = tuple(channels)
    cnn = CNNActually(input_resolution=input_resolution,
        identity_activations=identity_activations,
        channels=channels, classes=classes,
        feature_extractor=feature_extractor,
        batch_norm=batch_norm,
        debug=debug,
        mean=mean, std=std, div_by_255=div_by_255)
    cnn.to(memory_format=ch.channels_last)
    return cnn


class VGG(nn.Module):
    def __init__(self, 
        num_classes:int=None,
        depth:int=11,
        batch_norm:bool=False,
        mean: list[float] = IMAGENET_MEAN, 
        std: list[float] = IMAGENET_MEAN, 
        div_by_255:bool=False) -> None:
        super().__init__()
        self.nrmlz = Nrmlz(mean=mean, std=std, div_by_255=div_by_255)
        assert depth in [11,13,16,19], f'valid depths are [11,13,16,19] but got {depth}'
        bn = "_bn" if batch_norm else ""
        model_str = f'vgg{depth}{bn}'
        self.vgg: tvmods.VGG = getattr(tvmods, model_str)()
        if num_classes:
            in_feat: int = self.vgg.classifier[-1].in_features
            self.vgg.classifier[-1] = nn.Linear(in_feat, num_classes, bias=True)
    def forward(self, x:ch.Tensor):
        x = self.nrmlz(x)
        return self.vgg(x)

class VGGC(nn.Module):
    def __init__(self, 
        num_classes:int=None,
        depth:int=11,
        mean: list[float] = CIFAR_MEAN, 
        std: list[float] = CIFAR_STD, 
        div_by_255:bool=True) -> None:
        super().__init__()
        self.nrmlz = Nrmlz(mean=mean, std=std, div_by_255=div_by_255)
        assert depth in [11,13,16,19], f'valid depths are [11,13,16,19] but got {depth}'
        self.vgg: VGG_cifar = vgg_cifar_dict[depth]()
        if num_classes:
            in_feat: int = self.vgg.classifier[-1].in_features
            self.vgg.classifier[-1] = nn.Linear(in_feat, num_classes, bias=True)
    def forward(self, x:ch.Tensor):
        x = self.nrmlz(x)
        return self.vgg(x)


def imagenet_vgg(depth:int=11, pretrained:bool=False,
    mean: list[float] = IMAGENET_MEAN, 
    std: list[float] = IMAGENET_STD, 
    div_by_255:bool=False, batch_norm:bool=False):
    """returns pytorch VGG model for input depth
    model_depth must be in [11,13,16,19]
    batch_norm = False by default
    """
    assert depth in [11,13,16,19], f'valid depths are [11,13,16,19] but got {depth}'
    model = VGG(depth=depth, mean=mean, std=std, div_by_255=div_by_255, batch_norm=batch_norm)
    model = model.to(memory_format=ch.channels_last)
    return model

def imagenette_vgg(depth:int=11, pretrained:bool=False,
    mean: list[float] = IMAGENET_MEAN, 
    std: list[float] = IMAGENET_STD, 
    div_by_255:bool=False, batch_norm:bool=False):
    """returns  VGG model for input depth
    but with 10 classes
    model_depth must be in [11,13,16,19]
    batch_norm = False by default
    """
    assert depth in [11,13,16,19], f'valid depths are [11,13,16,19] but got {depth}'
    model = VGG(num_classes=10, depth=depth, mean=mean, std=std, 
        div_by_255=div_by_255, batch_norm=batch_norm)
    model = model.to(memory_format=ch.channels_last)
    return model

def vggc(depth:int=11, pretrained:bool=False,
    mean: list[float] = CIFAR_MEAN, 
    std: list[float] = CIFAR_STD, 
    div_by_255:bool=True):
    """returns cifar VGG model for input depth
    model_depth must be in [11,13,16,19]
    batch_norm = False by default
    """
    assert depth in [11,13,16,19], f'valid depths are [11,13,16,19] but got {depth}'
    model = VGGC(num_classes=10, depth=depth, mean=mean, std=std, 
        div_by_255=div_by_255)
    model = model.to(memory_format=ch.channels_last)
    return model

class Al(nn.Module):
    def __init__(self, 
        san_state_dict = None,
        mean: list[float] = LWN_MEAN, 
        std: list[float] = LWN_STD, 
        div_by_255:bool=False) -> None:
        super().__init__()
        self.nrmlz = Nrmlz(mean=mean, std=std, div_by_255=div_by_255)
        self.san: SmallAlexNet = SmallAlexNet()
        if san_state_dict != None:
            self.san.load_state_dict(san_state_dict)
        # get rid of l2 norm from alignment loss.
        self.san.blocks[-1] = self.san.blocks[-1][0]

    def forward(self, x:ch.Tensor):
        return self.san(self.nrmlz(x))

def al(pretrained:bool=False,
    san_state_dict = None,
    mean: list[float] = LWN_MEAN, 
    std: list[float] = LWN_STD, 
    div_by_255:bool=False):
    model = Al(san_state_dict=san_state_dict, 
        mean=mean, std=std, 
        div_by_255=div_by_255)
    model = model.to(memory_format=ch.channels_last)
    return model
