"""
Training models on natural and high pass filtered image datasets.
"""

import os
import torch as ch
from torch import nn
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from torch.utils.data import DataLoader
from ffcv import Loader
from typing import Callable
import pathlib as pa
import argparse as ap
from constants import results_dir
from datasets import cifardl, imagenetdl, dataloader_stats, imagenettedl
from models import rnc, rn, cnn_actually, mcnnk, imagenet_vgg, imagenette_vgg, vggc
from torchinfo import summary
from training import train
from utils import floating_string, chunkify


fsdir = results_dir / 'freq-sens'

default_train_args = {
    'lr': 1e-3,
    'num_epochs': 50,
}
default_sweep_args = {
    'runs': 5,
    'disable_progress_bar': True,
    'distributed': False,
    'device': ch.cuda.current_device()
}
default_model_args = {}

def decay_sweep(train_dataloader: DataLoader or Loader, 
    eval_dataloader: DataLoader or Loader,
    arch: Callable, decays: list[float], 
    sweep_dir: str or pa.Path,
    sweep_args: dict = default_sweep_args,
    train_args: dict = default_train_args,
    model_args: dict = default_model_args):
    """
    Train a collection of models at different levels of weight decay.
    """
    if isinstance(sweep_dir, str):
        sweep_dir = pa.Path(sweep_dir)
    sweep_dir: pa.Path
    # think this is unnecessary: train will make the dir if necessary
    # if not sweep_dir.exists():
    #     sweep_dir.mkdir(parents=True)
    print('storing results in:\n')
    print([sweep_dir / floating_string(l) for l in decays])
    for l in decays:
        for r in range(sweep_args['runs']):
            m: nn.Module = arch(pretrained=False, **model_args)
            if sweep_args['distributed']:
                m.to(sweep_args['rank'])
                m = DDP(m, device_ids=[sweep_args['rank']])
            train(model=m, train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                decay=l, 
                save_weights=True,
                save_interval=10,
                out_dir=sweep_dir / floating_string(l) / str(r),
                **train_args, **sweep_args)
            del m
            if sweep_args['distributed']:
                dist.barrier()

def depth_sweep(train_dataloader: DataLoader or Loader, 
    eval_dataloader: DataLoader or Loader,
    arch_family: Callable, depths: list[int], 
    sweep_dir: str or pa.Path,
    sweep_args: dict = default_sweep_args,
    train_args: dict = default_train_args,
    model_args: dict = default_model_args):
    """
    Train a bunch of models of varying depths. arch_family needs to be a
    function that eats an int, depth, and spits out an nn.Module, the net of
    desired depth.
    """
    if isinstance(sweep_dir, str):
        sweep_dir = pa.Path(sweep_dir)
    sweep_dir: pa.Path
    # see above
    # if not sweep_dir.exists():
    #     sweep_dir.mkdir(parents=True)
    print([sweep_dir / str(d) for d in depths])
    for d in depths:
        for r in range(sweep_args['runs']):
            m: nn.Module = arch_family(depth=d, **model_args)
            if sweep_args['distributed']:
                m.to(sweep_args['rank'])
                m = DDP(m, device_ids=[sweep_args['rank']])
            train(model=m, train_dataloader=train_dataloader, 
                eval_dataloader=eval_dataloader, 
                save_weights=True,
                save_interval=10,
                out_dir=sweep_dir / str(d) / str(r),
                **train_args, **sweep_args)
            del m

def cifar_rnc_decay_sweep(num_chunks:int=None, chunk:int=None, 
    sweep_args:dict=default_sweep_args, high_pass:float=None,
    untrained:bool = False):
    cfdl: dict[str, Loader] = cifardl(batch_size=512, num_workers=8, high_pass=high_pass)
    model_args = default_model_args
    if high_pass:
        mu, sigma = dataloader_stats(cfdl['train'])
        model_args = {
            'mean': mu,
            'std': sigma,
            'div_by_255': False
        }
    decays = list(10.0**np.linspace(-5, -2, 10))
    if num_chunks:
        decays = chunkify(decays, num_chunks=num_chunks, chunk=chunk)
    train_args = {
        'lr': 1e-1,
        'num_epochs': 500,
        'patience': 20,
        'min_lr' : 1e-6
    }
    sweep_dir=(fsdir / 'cifar_resnet_decay_sweep' if not high_pass else fsdir / f'cifar_resnet_decay_sweep_highpass_{floating_string(high_pass)}')
    if untrained:
        train_args['num_epochs'] = 0
        decays = [1e-3]
        sweep_dir = fsdir / 'cifar_resnet_decay_untrained'
    decay_sweep(train_dataloader=cfdl['train'],
        eval_dataloader=cfdl['test'],
        arch=rnc, decays=decays, 
        sweep_dir=sweep_dir, 
        train_args=train_args, sweep_args=sweep_args, model_args=model_args)

def cifar_mcnn_decay_sweep(num_chunks:int=None, chunk:int=None, 
    sweep_args:dict=default_sweep_args, high_pass:float=None,
    untrained:bool = False):
    cfdl: dict[str, Loader] = cifardl(batch_size=512, num_workers=8, high_pass=high_pass)
    model_args = default_model_args
    if high_pass:
        mu, sigma = dataloader_stats(cfdl['train'])
        model_args = {
            'mean': mu,
            'std': sigma,
            'div_by_255': False
        }
    decays = list(10.0**np.linspace(-5, -2, 10))
    if num_chunks:
        decays = chunkify(decays, num_chunks=num_chunks, chunk=chunk)
    train_args = {
        'lr': 1e-2,
        'num_epochs': 500,
        'patience': 20,
        'min_lr' : 1e-6
    }
    sweep_dir=(fsdir / 'cifar_mcnn_decay_sweep' if not high_pass else fsdir / f'cifar_mcnn_decay_sweep_highpass_{floating_string(high_pass)}')
    if untrained:
        train_args['num_epochs'] = 0
        decays = [1e-3]
        sweep_dir = fsdir / 'cifar_mcnn_decay_untrained'
    decay_sweep(train_dataloader=cfdl['train'],
        eval_dataloader=cfdl['test'],
        arch=mcnnk, decays=decays, 
        sweep_dir=sweep_dir,  
        train_args=train_args, sweep_args=sweep_args, model_args=model_args)

def cifar_cnna_decay_sweep(num_chunks:int=None, chunk:int=None, 
    sweep_args:dict=default_sweep_args, high_pass:float=None,
    untrained:bool = False):
    cfdl: dict[str, Loader] = cifardl(batch_size=1024, num_workers=8, high_pass=high_pass)
    model_args = default_model_args
    if high_pass:
        mu, sigma = dataloader_stats(cfdl['train'])
        model_args = {
            'mean': mu,
            'std': sigma,
            'div_by_255': False,
            'feature_extractor': 'flatten'
        }
    # decays = list(10.0**np.linspace(-5, -1, 10))
    # dialing down to get more runs
    decays = list(10.0**np.linspace(-5, -1, 4))
    if num_chunks:
        decays = chunkify(decays, num_chunks=num_chunks, chunk=chunk)
    train_args = {
        'lr': 1e-4,
        'num_epochs': 500,
        'patience': 20,
        'min_lr': 1e-6,
    }
    sweep_dir = (fsdir / 'cifar_cnna_decay_sweep' if not high_pass else fsdir / f'cifar_cnna_decay_sweep_highpass_{floating_string(high_pass)}')
    if untrained:
        train_args['num_epochs'] = 0
        decays = [1e-3]
        sweep_dir = fsdir / 'cifar_cnna_decay_untrained'
    decay_sweep(train_dataloader=cfdl['train'],
        eval_dataloader=cfdl['test'],
        arch=cnn_actually, decays=decays,  
        sweep_dir=sweep_dir, 
        train_args=train_args, sweep_args=sweep_args, model_args=model_args)
    
def cifar_linear_cnn_decay_sweep(num_chunks:int=None, chunk:int=None, 
    sweep_args:dict=default_sweep_args, high_pass:float=None,
    untrained:bool = False):
    cfdl: dict[str, Loader] = cifardl(batch_size=1024, num_workers=8, high_pass=high_pass)
    model_args = default_model_args
    if high_pass:
        mu, sigma = dataloader_stats(cfdl['train'])
        model_args = {
            'mean': mu,
            'std': sigma,
            'div_by_255': False,
            'feature_extractor': 'flatten',
            'identity_activations': True
        }
    # decays = list(10.0**np.linspace(-5, -1, 10))
    # dialing down to get more runs
    decays = list(10.0**np.linspace(-5, -1, 4))
    if num_chunks:
        decays = chunkify(decays, num_chunks=num_chunks, chunk=chunk)
    train_args = {
        'lr': 1e-4,
        'num_epochs': 500,
        'patience': 20,
        'min_lr': 1e-6,
    }
    sweep_dir = (fsdir / 'cifar_linear_cnn_decay_sweep' if not high_pass else fsdir / f'cifar_linear_cnn_decay_sweep_highpass_{floating_string(high_pass)}')
    if untrained:
        train_args['num_epochs'] = 0
        decays = [1e-3]
        sweep_dir = fsdir / 'cifar_linear_cnn_decay_untrained'
    decay_sweep(train_dataloader=cfdl['train'],
        eval_dataloader=cfdl['test'],
        arch=cnn_actually, decays=decays,  
        sweep_dir=sweep_dir, 
        train_args=train_args, sweep_args=sweep_args, model_args=model_args)

def cifar_rnc_depth_sweep(num_chunks:int=None, chunk:int=None, 
    sweep_args:dict=default_sweep_args, high_pass:float=None,
    untrained:bool = False):
    cfdl: dict[str, Loader] = cifardl(batch_size=512, num_workers=8, high_pass=high_pass,)
    model_args = default_model_args
    if high_pass:
        mu, sigma = dataloader_stats(cfdl['train'])
        model_args = {
            'mean': mu,
            'std': sigma,
            'div_by_255': False
        }
    depths = [9, 20, 56]
    if num_chunks:
        depths = chunkify(depths, num_chunks=num_chunks, chunk=chunk)
    train_args = {
        'lr': 1e-1,
        # TODO: why wasn't this 500??
        'num_epochs': 200,
        'patience': 20,
        'decay': 1e-5
    }
    sweep_dir=(fsdir / 'cifar_resnet_depth_sweep' if not high_pass else fsdir / f'cifar_resnet_depth_sweep_highpass_{floating_string(high_pass)}')
    if untrained:
        train_args['num_epochs'] = 0
        sweep_dir = fsdir / 'cifar_resnet_depth_untrained'
    depth_sweep(train_dataloader=cfdl['train'],
        eval_dataloader=cfdl['test'],
        arch_family=rnc, depths=depths,  
        sweep_dir=sweep_dir,  
        train_args=train_args, sweep_args=sweep_args, model_args=model_args)

def cifar_cnna_depth_sweep(num_chunks:int=None, chunk:int=None, 
    sweep_args:dict=default_sweep_args, high_pass:float=None, 
    untrained:bool = False):
    cfdl: dict[str, Loader] = cifardl(batch_size=1024, num_workers=8, high_pass=high_pass)
    model_args = default_model_args
    if high_pass:
        mu, sigma = dataloader_stats(cfdl['train'])
        model_args = {
            'mean': mu,
            'std': sigma,
            'div_by_255': False,
            'feature_extractor': 'flatten'
        }
    # dialing down to get more runs
    # depths = list(range(1,10))
    depths = [1,2,4,8]
    if num_chunks:
        depths = chunkify(depths, num_chunks=num_chunks, chunk=chunk)
    train_args = {
        'lr': 1e-4,
        'num_epochs': 500,
        'patience': 20,
        'min_lr': 1e-6,
        'decay': 1e-5
    }
    sweep_dir = (fsdir / 'cifar_cnna_depth_sweep' if not high_pass else fsdir / f'cifar_cnna_depth_sweep_highpass_{floating_string(high_pass)}')
    if untrained:
        train_args['num_epochs'] = 0
        sweep_dir = fsdir / 'cifar_cnna_depth_untrained'
    depth_sweep(train_dataloader=cfdl['train'],
        eval_dataloader=cfdl['test'],
        arch_family=cnn_actually, depths=depths,  
        sweep_dir=sweep_dir,  
        train_args=train_args, sweep_args=sweep_args, model_args=model_args)
    
def cifar_linear_cnn_depth_sweep(num_chunks:int=None, chunk:int=None, 
    sweep_args:dict=default_sweep_args, high_pass:float=None, 
    untrained:bool = False):
    cfdl: dict[str, Loader] = cifardl(batch_size=1024, num_workers=8, high_pass=high_pass)
    model_args = default_model_args
    if high_pass:
        mu, sigma = dataloader_stats(cfdl['train'])
        model_args = {
            'mean': mu,
            'std': sigma,
            'div_by_255': False,
            'feature_extractor': 'flatten',
            'identity_activations': True,
        }
    # dialing down to get more runs
    # depths = list(range(1,10))
    depths = [1,2,4,8]
    if num_chunks:
        depths = chunkify(depths, num_chunks=num_chunks, chunk=chunk)
    train_args = {
        'lr': 1e-4,
        'num_epochs': 500,
        'patience': 20,
        'min_lr': 1e-6,
        'decay': 1e-5
    }
    sweep_dir = (fsdir / 'cifar_linear_cnn_depth_sweep' if not high_pass else fsdir / f'cifar_linear_cnn_depth_sweep_highpass_{floating_string(high_pass)}')
    if untrained:
        train_args['num_epochs'] = 0
        sweep_dir = fsdir / 'cifar_linear_cnn_depth_untrained'
    depth_sweep(train_dataloader=cfdl['train'],
        eval_dataloader=cfdl['test'],
        arch_family=cnn_actually, depths=depths,  
        sweep_dir=sweep_dir,  
        train_args=train_args, sweep_args=sweep_args, model_args=model_args)


def vgg_decay_sweep(num_chunks:int=None, chunk:int=None,
    sweep_args:dict=default_sweep_args,  high_pass:float=None,
    untrained:bool = False):
    cfdl: dict[str, Loader] = cifardl(batch_size=256, num_workers=8, high_pass=high_pass)
    model_args = default_model_args
    if high_pass:
        mu, sigma = dataloader_stats(cfdl['train'])
        model_args = {
            'mean': mu,
            'std': sigma,
            'div_by_255': False
        }
    decays = list(10.0**np.linspace(-5, -2, 10))
    if num_chunks:
        decays = chunkify(decays, num_chunks=num_chunks, chunk=chunk)
    train_args = {
        'lr': 1e-2,
        'num_epochs': 500,
        'patience': 20,
        'min_lr' : 1e-6
    }
    sweep_dir=(fsdir / 'cifar_vgg_decay_sweep' if not high_pass else fsdir / f'cifar_vgg_decay_sweep_highpass_{floating_string(high_pass)}')
    if untrained:
        train_args['num_epochs'] = 0
        decays = [1e-3]
        sweep_dir = fsdir / 'cifar_vgg_decay_untrained'
    decay_sweep(train_dataloader=cfdl['train'],
        eval_dataloader=cfdl['test'],
        arch=vggc, decays=decays, 
        sweep_dir=sweep_dir,
        train_args=train_args, sweep_args=sweep_args, model_args=model_args)

def vgg_depth_sweep(num_chunks:int=None, chunk:int=None, 
    sweep_args:dict=default_sweep_args, high_pass:float=None, 
    untrained:bool = False):
    cfdl: dict[str, Loader] = cifardl(batch_size=256, num_workers=8, high_pass=high_pass)
    model_args = default_model_args
    if high_pass:
        mu, sigma = dataloader_stats(cfdl['train'])
        model_args = {
            'mean': mu,
            'std': sigma,
            'div_by_255': False
        }
    depths = [11,13,16,19]
    if num_chunks:
        depths = chunkify(depths, num_chunks=num_chunks, chunk=chunk)
    train_args = {
        'lr': 1e-2,
        'num_epochs': 500,
        'patience': 20,
        'min_lr' : 1e-6,
        'decay': 1e-5
    }
    sweep_dir=(fsdir / 'cifar_vgg_depth_sweep' if not high_pass else fsdir / f'cifar_vgg_depth_sweep_highpass_{floating_string(high_pass)}')
    if untrained:
        train_args['num_epochs'] = 0
        sweep_dir = fsdir / 'cifar_vgg_depth_untrained'
    depth_sweep(train_dataloader=cfdl['train'],
        eval_dataloader=cfdl['test'],
        arch_family=vggc, depths=depths,  
        sweep_dir=sweep_dir,
        train_args=train_args, sweep_args=sweep_args, model_args=model_args)

def imagenette_vgg_decay_sweep(num_chunks:int=None, chunk:int=None,
    sweep_args:dict=default_sweep_args,  high_pass:float=None,
    untrained:bool = False):
    imdl: dict[Loader] = imagenettedl(batch_size=32, num_workers=4, high_pass=high_pass, distributed=sweep_args['distributed'], device=sweep_args['rank'])
    model_args = default_model_args
    if high_pass:
        mu, sigma = dataloader_stats(imdl['train'])
        model_args = {
            'mean': mu,
            'std': sigma,
            'div_by_255': False
        }
    # dialing down to get more runs
    # decays = list(10.0**np.linspace(-5, -2, 10))
    decays = list(10.0**np.linspace(-5, -2, 3))
    if num_chunks:
        decays = chunkify(decays, num_chunks=num_chunks, chunk=chunk)
    train_args = {
        'lr': 1e-2*(32/256),
        'num_epochs': 500,
        'patience': 20,
        'min_lr' : 1e-6
    }
    sweep_dir=(fsdir / 'imagenette_vgg_decay_sweep' if not high_pass else fsdir / f'imagenette_vgg_decay_sweep_highpass_{floating_string(high_pass)}')
    if untrained:
        train_args['num_epochs'] = 0
        sweep_dir = fsdir / 'imagenette_vgg_decay_untrained'
    decay_sweep(train_dataloader=imdl['train'],
        eval_dataloader=imdl['test'],
        arch=imagenette_vgg, decays=decays, 
        sweep_dir=sweep_dir,
        train_args=train_args, sweep_args=sweep_args, model_args=model_args)

def imagenette_vgg_depth_sweep(num_chunks:int=None, chunk:int=None, 
    sweep_args:dict=default_sweep_args, high_pass:float=None, 
    untrained:bool = False):
    imdl: dict[Loader] = imagenettedl(batch_size=32, num_workers=4, high_pass=high_pass, distributed=sweep_args['distributed'], device=sweep_args['rank'])
    model_args = default_model_args
    if high_pass:
        mu, sigma = dataloader_stats(imdl['train'])
        model_args = {
            'mean': mu,
            'std': sigma,
            'div_by_255': False
        }
    depths = [11,13,16,19]
    if num_chunks:
        depths = chunkify(depths, num_chunks=num_chunks, chunk=chunk)
    train_args = {
        'lr': 1e-2*(32/256),
        'num_epochs': 500,
        'patience': 20,
        'min_lr' : 1e-6,
        'decay': 1e-5
    }
    sweep_dir=(fsdir / 'imagenette_vgg_depth_sweep' if not high_pass else fsdir / f'imagenette_vgg_depth_sweep_highpass_{floating_string(high_pass)}')
    if untrained:
        train_args['num_epochs'] = 0
        sweep_dir = fsdir / 'imagenette_vgg_depth_untrained'
    depth_sweep(train_dataloader=imdl['train'],
        eval_dataloader=imdl['test'],
        arch_family=imagenette_vgg, depths=depths,  
        sweep_dir=sweep_dir,
        train_args=train_args, sweep_args=sweep_args, model_args=model_args)

if __name__=='__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('--sweep', type=str, default=None, help='type of sweep to run')
    parser.add_argument('--runs', type=int, default=5, help='number of runs per hyperparam')
    parser.add_argument('--high_pass', type=float, default=None, help='high pass filter ratio')
    parser.add_argument('--num_chunks', type=int, default=None, help='number of chunks')
    parser.add_argument('--chunk', type=int, default=None, help='chunk')
    parser.add_argument('--progress', action='store_true', help='bars?')
    parser.add_argument('--distributed', action='store_true', help='multi-gpu?')
    parser.add_argument('--untrained', action='store_true', help='skip training??')
    args = parser.parse_args()
    sweep_args = { 
        'disable_progress_bar': (not args.progress),
        'distributed': args.distributed,
        'runs': args.runs
    }
    if args.distributed:
        rank = int(os.environ["LOCAL_RANK"])
        ws = int(os.environ['WORLD_SIZE'])
        print(f'rank {rank} of worldsize {ws}')
        ch.cuda.set_device(rank)   
        dist.init_process_group(backend="nccl")
        sweep_args['rank'] = rank
        sweep_args['device'] = None
    else:
        device = ch.device('cuda:0')
        sweep_args['device'] = device
    if args.sweep == 'decay_rnc':
        cifar_rnc_decay_sweep(num_chunks=args.num_chunks, 
            chunk=args.chunk, sweep_args=sweep_args, high_pass=args.high_pass,
            untrained=args.untrained)
    elif args.sweep == 'decay_mcnn':
        cifar_mcnn_decay_sweep(num_chunks=args.num_chunks, 
            chunk=args.chunk, sweep_args=sweep_args, high_pass=args.high_pass,
            untrained=args.untrained)
    elif args.sweep == 'decay_cnna':
        sweep_args['runs'] = 5
        cifar_cnna_decay_sweep(num_chunks=args.num_chunks, 
            chunk=args.chunk, sweep_args=sweep_args, high_pass=args.high_pass,
            untrained=args.untrained)
    elif args.sweep == 'decay_linear_cnn':
        sweep_args['runs'] = 5
        cifar_linear_cnn_decay_sweep(num_chunks=args.num_chunks, 
            chunk=args.chunk, sweep_args=sweep_args, high_pass=args.high_pass,
            untrained=args.untrained)
    elif args.sweep == 'depth_rnc':
        cifar_rnc_depth_sweep(num_chunks=args.num_chunks, 
            chunk=args.chunk, sweep_args=sweep_args, high_pass=args.high_pass,
            untrained=args.untrained)
    elif args.sweep == 'depth_cnna':
        sweep_args['runs'] = 5
        cifar_cnna_depth_sweep(num_chunks=args.num_chunks, 
            chunk=args.chunk, sweep_args=sweep_args, high_pass=args.high_pass,
            untrained=args.untrained)
    elif args.sweep == 'depth_linear_cnn':
        sweep_args['runs'] = 5
        cifar_linear_cnn_depth_sweep(num_chunks=args.num_chunks, 
            chunk=args.chunk, sweep_args=sweep_args, high_pass=args.high_pass,
            untrained=args.untrained)
    # elif args.sweep == 'depth_cnna_exp_width':
    #     # can only afford 1 run for now :( 
    #     sweep_args['runs'] = 1
    #     cifar_cnna_depth_sweep_exp_width(num_chunks=args.num_chunks, 
    #         chunk=args.chunk, sweep_args=sweep_args, high_pass=args.high_pass,)
    elif args.sweep == 'cnna_train':
        cnna_train(progress=args.progress)
    elif args.sweep == 'decay_vgg_imagenette':
        imagenette_vgg_decay_sweep(num_chunks=args.num_chunks, 
            chunk=args.chunk, sweep_args=sweep_args, high_pass=args.high_pass)
    elif args.sweep == 'depth_vgg_imagenette':
        imagenette_vgg_depth_sweep(num_chunks=args.num_chunks, 
            chunk=args.chunk, sweep_args=sweep_args, high_pass=args.high_pass,
            untrained=args.untrained)
    elif args.sweep == 'decay_vgg':
        vgg_decay_sweep(num_chunks=args.num_chunks, 
            chunk=args.chunk, sweep_args=sweep_args, high_pass=args.high_pass,
            untrained=args.untrained)
    elif args.sweep == 'depth_vgg':
        vgg_depth_sweep(num_chunks=args.num_chunks, 
            chunk=args.chunk, sweep_args=sweep_args, high_pass=args.high_pass,
            untrained=args.untrained)
    # elif args.sweep == 'decay_vgg_imagenet':
    #     imagenet_vgg_decay_sweep(num_chunks=args.num_chunks, 
    #         chunk=args.chunk, sweep_args=sweep_args)
    # elif args.sweep == 'depth_vgg_imagenet':
    #     imagenet_vgg_depth_sweep(num_chunks=args.num_chunks, 
    #         chunk=args.chunk, sweep_args=sweep_args)
if args.distributed:
    dist.init_process_group(backend="nccl")





