"""
test frequency sensitivity of CNNs, compute frequency distributions of datasets
"""

import torch as ch
from torch import nn, fft
from torch.autograd import functional as agfun
from torch import linalg as la
from torch.utils.data import DataLoader
import numpy as np
from ffcv import Loader
from typing import Callable
from tqdm.auto import tqdm
from constants import results_dir
from utils import chunkify, floating_string
from models import cifar_nrmlz, inet_nrmlz, rnc, mcnnk, cnn_actually, al, vggc, imagenette_vgg, Nrmlz
from datasets import cifardl, dataloader_stats, imagenettedl, lwndl
import argparse as ap
from plotting import *

fsdir = results_dir / 'freq-sens'
smdir = fsdir / 'sensitivity-maps'
covdir = fsdir / 'covariances'

def fulltranspose(x:ch.Tensor) -> ch.Tensor:
    return x.permute(ch.Size(reversed(list(range(len(x.shape))))))

def average_smooth(*xs: list[ch.Tensor], smooth:int = 1, datatype: np.float64):
    t = np.arange(-smooth, smooth+1)
    smther = np.ones(t.shape)
    smther /= smther.sum()
    smther = smther.astype(datatype)
    return  [np.convolve(x, smther, mode='valid') for x in xs]

def fourier_grad_sens(dataloader: DataLoader or Loader,
    model:nn.Module, progress:bool = False,
    loss: nn.Module = None, 
    device: int = ch.cuda.current_device()):
    model.eval()
    model.to(device)
    # don't want to worry about mentally mapping shapes :) 
    input_shape = None
    for x, y in dataloader:
        if isinstance(x, list) or isinstance(x, tuple):
            x = ch.cat(x, dim=0)
        input_shape = x[0].shape
        break
    xtest = ch.zeros(input_shape)
    # w/o abs, sens_map is complex which causes pain later
    xhattest = fft.fft2(xtest, norm='ortho')
    sens_maps = {
            k: [ch.zeros_like(xhattest.abs()), ch.zeros_like(xhattest.abs())]
            for k in ['meannorm', 'rms']
        }
    del xtest, xhattest
    loop = dataloader
    if progress:
        loop = tqdm(dataloader)
    model.to(device)
    for x, y in loop:
        # hack deal with the lwn unsupervised dataloader
        if isinstance(x, list) or isinstance(x, tuple):
            x = ch.cat(x, dim=0)
        x, y = x.to(device), y.to(device)
        xhat: ch.Tensor = fft.fft2(x, norm='ortho') 
        xhat.requires_grad = True
        xhat.grad = None
        # again abs ensures correct dtype later on
        xhat_mn_accum = ch.zeros_like(xhat.abs())
        xhat_rms_accum = ch.zeros_like(xhat.abs())
        for i in range(xhat.shape[0]):
            if loss:
                l: Callable= lambda v:  loss(model(fft.ifft2(v.unsqueeze(0), norm='ortho').real), y[i].unsqueeze(0))
            else:
                l: Callable= lambda v:  model(fft.ifft2(v.unsqueeze(0), norm='ortho').real)
            jac = agfun.jacobian(l, inputs=xhat[i])
            xhat_mn_accum[i] = ch.norm(jac.abs(), dim = tuple(range(len(jac.shape[:-3]))))
            xhat_rms_accum[i] = ch.mean(jac.abs()**2, dim=tuple(range(len(jac.shape[:-3]))))
        sens_maps['meannorm'][0] += ch.mean(xhat_mn_accum, dim=0).cpu()
        sens_maps['meannorm'][1] += ch.var(xhat_mn_accum, dim=0).cpu()
        sens_maps['rms'][0] += ch.mean(xhat_rms_accum, dim=0).cpu()
        sens_maps['rms'][1] += ch.var(xhat_rms_accum, dim=0).cpu()
    sens_maps = {
        k: [x/len(dataloader) for x in v]
        for k, v in sens_maps.items()
    }
    # last annoying step: get standard deviations (rather than variance) and use
    # delta method to propagate thru R of RMS
    for k in sens_maps.keys():
        sens_maps[k][1] = ch.sqrt(sens_maps[k][1])
    sens_maps['rms'][0] = ch.sqrt(sens_maps['rms'][0])
    sens_maps['rms'][1] = ((1/2)*(sens_maps['rms'][0])**(-1/2))*sens_maps['rms'][1]
    return sens_maps

def empirical_power(dataloader: DataLoader or Loader,
    center:bool=False,
    device: int = ch.cuda.current_device(), progress:bool=False):
    MEAN, STD = dataloader_stats(dataloader)
    nrmlz: Nrmlz = Nrmlz(mean=MEAN, std=STD, div_by_255=False)
    # don't want to worry about mentally mapping shapes :) 
    input_shape = None
    for x, _ in dataloader:
        if isinstance(x, list) or isinstance(x, tuple):
            x = ch.cat(x, dim=0)
        input_shape = x[0].shape
        break
    xtest = ch.zeros(input_shape)
    xhattest = fft.fft2(xtest)
    # abs makes power_map real
    mu = ch.zeros_like(xhattest)
    power_map = ch.zeros_like(xhattest.abs())
    # print("sens_map.shape:",sens_map.shape)
    del xtest, xhattest
    loop = dataloader
    if progress:
        loop = tqdm(dataloader)
    if center:
        # compute mean first
        for x, _ in loop:
            if isinstance(x, list) or isinstance(x, tuple):
                x = ch.cat(x, dim=0)
            x = x.to(device)
            x = nrmlz(x)
            xhat: ch.Tensor = fft.fft2(x, norm='ortho') 
            mu += ch.mean(xhat, dim=0).cpu()
        mu /= len(dataloader)
    # now compute diag of covar matrix (closest comparison w/ theory)
    for x, _ in loop:
        if isinstance(x, list) or isinstance(x, tuple):
            x = ch.cat(x, dim=0)
        x = x.to(device)
        x = nrmlz(x)
        mu = mu.to(device)
        xhat: ch.Tensor = fft.fft2(x, norm='ortho') 
        xhat -= mu
        power_map += ch.mean(xhat.abs()**2, dim=0).cpu()
    power_map /= len(dataloader)
    power_map = ch.sqrt(power_map)
    return power_map

def empirical_xtx(dataloader: DataLoader or Loader,
    center: bool=False,
    device: int = ch.cuda.current_device(), progress:bool=False):
    MEAN, STD = dataloader_stats(dataloader)
    nrmlz: Nrmlz = Nrmlz(mean=MEAN, std=STD, div_by_255=False)
    # don't want to worry about mentally mapping shapes :) 
    input_shape =  None
    for x, _ in dataloader:
        if isinstance(x, list) or isinstance(x, tuple):
            x = ch.cat(x, dim=0)
        input_shape = x[0].shape
        break
    xtest = ch.zeros(input_shape)
    xhattest = fft.fft2(xtest)
    mu = ch.zeros_like(xhattest).to(device)
    xtx: ch.Tensor = ch.zeros(list(reversed([d for d in xhattest.shape])) + list(xhattest.shape))
    xtx = xtx.to(ch.complex32).to(device)
    del xtest, xhattest
    loop = dataloader
    if progress:
        loop = tqdm(dataloader)
    if center:
        # again, compute mean first
        for x, _ in loop:
            if isinstance(x, list) or isinstance(x, tuple):
                x = ch.cat(x, dim=0)
            x = x.to(device)
            x = nrmlz(x)
            xhat: ch.Tensor = fft.fft2(x, norm='ortho') 
            mu += ch.mean(xhat, dim=0)
        mu /= len(dataloader)
    for x, _ in loop:
        if isinstance(x, list) or isinstance(x, tuple):
            x = ch.cat(x, dim=0)
        x = x.to(device)
        x = nrmlz(x)
        xhat: ch.Tensor = fft.fft2(x, norm='ortho') 
        xhat -= mu
        xtx += ch.tensordot(fulltranspose(xhat).conj(), xhat, dims=1)/xhat.shape[0]
    
    xtx /= len(dataloader)
    return xtx.cpu()

def covrows(xtx: np.ndarray, progress:bool=False, debug:bool=True, smooth:int=None,
    norm_first:bool=False, modular:bool=False) -> tuple[ch.Tensor]:
    W, H, C = xtx.shape[:3]
    Cp, Hp, Wp = xtx.shape[3:]
    assert (C, H, W) == (Cp, Hp, Wp), f'need a symmetric xtx'
    mgw, mgh, mgc, mgcp, mghp, mgwp = np.meshgrid(*[np.arange(n) for n in xtx.shape], indexing='ij')
    def modular_dist(x: np.ndarray, y: np.ndarray, m: int):
        delta = np.mod(x - y, m)
        delta = np.min(np.stack([delta, m-delta], axis=0), axis=0)
        return np.abs(delta)
    if modular:
        dists = np.sqrt((modular_dist(mgw, mgwp, W))**2 + (modular_dist(mgh, mghp, H))**2 + (mgc - mgcp)**2)
    else:
        dists = np.sqrt((mgw - mgwp)**2 + (mgh - mghp)**2 + (mgc - mgcp)**2)
    dists_range = np.unique(dists.reshape((-1,)))
    # paranoia
    dists_range = np.sort(dists_range)
    datatype: np.dtype = None
    if norm_first:
        xtx = np.abs(xtx)
        datatype = np.float64
    else:
        datatype = np.complex64
    ans = np.zeros_like(dists_range)
    ans = ans.astype(datatype)
    bin_sizes = np.zeros_like(dists_range)
    loop = enumerate(dists_range)
    if progress:
        loop = tqdm(loop)
    for i, d in loop:
        mask = (dists == d).astype(datatype)
        mask_sum = mask.sum().astype(datatype)
        ans[i] = (mask*xtx).sum()/mask_sum
        bin_sizes[i] = mask_sum
        print((mask*xtx).sum(), mask_sum, ans[i])
    if smooth:
        t = np.arange(-smooth, smooth+1)
        # smther = np.exp(-t**2/2)
        smther = np.ones(t.shape)
        smther /= smther.sum()
        smther = smther.astype(datatype)
        ans, dists_range = np.convolve(ans, smther, mode='valid'), np.convolve(dists_range, smther, mode='valid')
    if debug:
        return dists_range, ans, bin_sizes
    else:
        return dists_range, ans

def fgs_decay_sweep(
    dataloader: DataLoader or Loader,
    arch: Callable, 
    sweep_dir: str or pa.Path,
    plot:bool=False, plot_dir:str or pa.Path = None,
    out_file: pa.Path = None,
    paranoid:bool = True,
    plot_suffix:str = None,
    device: ch.device or int = ch.cuda.current_device(),
    progress:bool=False,
    high_pass:float=None,
    lwn_hack:bool=False,
    dist_hack:bool=False,
    linear_hack: bool=False):
    model_args: dict = {}
    if high_pass:
        mu, sigma = dataloader_stats(dataloader)
        model_args = {
            'mean': mu,
            'std': sigma,
            'div_by_255': False
        }
    if arch == cnn_actually:
        model_args['feature_extractor'] = 'flatten'
        if linear_hack:
            model_args['identity_activations'] = True
    if not isinstance(sweep_dir, pa.Path):
        sweep_dir = pa.Path(sweep_dir)
    if plot_dir and not isinstance(plot_dir, pa.Path):
        plot_dir = pa.Path(plot_dir)
    sensitivity_maps = {}
    for f in sweep_dir.iterdir():
        l = floating_string(f.parts[-1])
        sensitivity_maps[l] = {}
        for g in f.iterdir():
            r = int(g.parts[-1])
            print(l, r, g)
            if lwn_hack:
                wts = ch.load(g / 'best_wts.pt')
                m: nn.Module = arch(pretrained=False, san_state_dict = wts,  **model_args)
            elif dist_hack:
                m: nn.Module = arch(pretrained=False, **model_args)
                wts = ch.load(g / 'best_wts.pt')
                wts = {'.'.join(k.split('.')[1:]): v for k, v in wts.items()}
                m.load_state_dict(wts)
            else:
                m: nn.Module = arch(pretrained=False, **model_args)
                wts = ch.load(g / 'best_wts.pt')
                m.load_state_dict(wts)
            m.cpu()
            sensitivity_maps[l][r] = fourier_grad_sens(dataloader=dataloader,
                model=m, device=device, progress=progress)
            if plot:
                fig, _ = frequency_image(sensitivity_maps[l][r], logparam=True)
                fig.savefig(plot_dir / f'{f.parts[-1]}_{g.parts[-1]}_{plot_suffix}.png')
            if paranoid and out_file:
                with open(out_file, 'wb') as of:
                    pkl.dump(sensitivity_maps, of)
    if out_file:
        with open(out_file, 'wb') as of:
            pkl.dump(sensitivity_maps, of)
    return sensitivity_maps

def fgs_depth_sweep(
    dataloader: DataLoader or Loader,
    arch_family: Callable,
    sweep_dir: str or pa.Path,
    plot:bool=False, plot_dir:str or pa.Path = None,
    out_file: pa.Path = None,
    paranoid:bool = True,
    plot_suffix:str = None,
    device: ch.device or int = ch.cuda.current_device(),
    progress:bool=False,
    high_pass: float = None,
    dist_hack:bool=False,
    linear_hack: bool=False):
    model_args: dict = {}
    if high_pass:
        mu, sigma = dataloader_stats(dataloader)
        model_args = {
            'mean': mu,
            'std': sigma,
            'div_by_255': False
        }
    if arch_family == cnn_actually:
        model_args['feature_extractor'] = 'flatten'
        if linear_hack:
            model_args['identity_activations'] = True
    if not isinstance(sweep_dir, pa.Path):
        sweep_dir = pa.Path(sweep_dir)
    if plot_dir and not isinstance(plot_dir, pa.Path):
        plot_dir = pa.Path(plot_dir)
    sensitivity_maps = {}
    print(list(sweep_dir.iterdir()))
    for f in sweep_dir.iterdir():
        d = int(f.parts[-1])
        sensitivity_maps[d] = {}
        for g in f.iterdir():
            r = int(g.parts[-1])
            print(d, r, g)
            if dist_hack:
                m: nn.Module = arch_family(depth=d, pretrained=False, **model_args)
                wts = ch.load(g / 'best_wts.pt')
                wts = {'.'.join(k.split('.')[1:]): v for k, v in wts.items()}
                m.load_state_dict(wts)
            else:
                m: nn.Module = arch_family(depth=d, pretrained=False, **model_args)
                wts = ch.load(g / 'best_wts.pt')
                m.load_state_dict(wts)
            sensitivity_maps[d][r] = fourier_grad_sens(dataloader=dataloader,
                model=m, device=device, progress=progress)
            if plot:
                fig, _ = frequency_image(sensitivity_maps[d][r], logparam=True)
                fig.savefig(plot_dir / f'{f.parts[-1]}_{g.parts[-1]}_{plot_suffix}.png')
            if paranoid and out_file:
                with open(out_file, 'wb') as of:
                    pkl.dump(sensitivity_maps, of)
    if out_file:
        with open(out_file, 'wb') as of:
            pkl.dump(sensitivity_maps, of)
    return sensitivity_maps
  
if __name__=='__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('--sweep', type=str, default=None, help='type of sweep to run')
    parser.add_argument('--phase', type=str, default=None, help='train, test or both')
    parser.add_argument('--subsample', type=int, default=None, 
        help='number of subsamples from the underlying dataset')
    parser.add_argument('--high_pass', type=float, default=None, help='high pass filter ratio')
    # parser.add_argument('--num_chunks', type=int, default=None, help='number of chunks')
    # parser.add_argument('--chunk', type=int, default=None, help='chunk')
    parser.add_argument('--progress', action='store_true', help='bars?')
    parser.add_argument('--untrained', action='store_true', help='skip training??')
    args = parser.parse_args()
    # TODO: need to make sure untrained is actually getting thru
    device = ch.device('cuda:0')
    if args.phase == None:
        phases = ['test', 'train']
    else:
        phases = [args.phase]
    if args.sweep == 'decay_rnc':
        cfdl = cifardl(batch_size=64, num_workers=8, half=False, 
            subsample=args.subsample, high_pass=args.high_pass)
        for phase, dl in cfdl.items():
            if phase not in phases:
                continue
            sweep_dir = fsdir / ('cifar_resnet_decay_untrained' if args.untrained else 'cifar_resnet_decay_sweep')
            out_file=smdir / f'fgs_decay_sweep_cifar_rnc_{phase}.pkl'
            if args.high_pass:
                sweep_dir = sweep_dir.parent / (str(sweep_dir.parts[-1]) + f'_highpass_{floating_string(args.high_pass)}')
                out_file = out_file.parent / (str(out_file.stem) + f'_highpass_{floating_string(args.high_pass)}.pkl')
            elif args.untrained:
                out_file = out_file.parent / (str(out_file.stem) + f'_untrained.pkl')
            sm = fgs_decay_sweep(dataloader=dl,
                arch=rnc, sweep_dir = sweep_dir,
                plot=False, plot_dir=pa.Path('./plots/fgs_decay_sweep_cifar_rnc'), plot_suffix=phase,
                out_file=out_file,
                progress=args.progress) 
    elif args.sweep == 'decay_mcnn':
        cfdl = cifardl(batch_size=64, num_workers=8, half=False, 
            subsample=args.subsample, high_pass=args.high_pass)
        for phase, dl in cfdl.items():
            if phase not in phases:
                continue
            sweep_dir = fsdir / ('cifar_mcnn_decay_untrained' if args.untrained else 'cifar_mcnn_decay_sweep')
            out_file=smdir / f'fgs_decay_sweep_cifar_mcnn_{phase}.pkl'
            if args.high_pass:
                sweep_dir = sweep_dir.parent / (str(sweep_dir.parts[-1]) + f'_highpass_{floating_string(args.high_pass)}')
                out_file = out_file.parent / (str(out_file.stem) + f'_highpass_{floating_string(args.high_pass)}.pkl')
            elif args.untrained:
                out_file = out_file.parent / (str(out_file.stem) + f'_untrained.pkl')
            sm = fgs_decay_sweep(dataloader=dl,
                arch=mcnnk, sweep_dir = sweep_dir, 
                plot=False, plot_dir=pa.Path('./plots/fgs_decay_sweep_cifar_mcnn'), plot_suffix=phase, 
                out_file=out_file, 
                progress=args.progress) 
    elif args.sweep == 'decay_cnna':
        cfdl = cifardl(batch_size=64, num_workers=8, half=False, 
            subsample=args.subsample, high_pass=args.high_pass)
        for phase, dl in cfdl.items():
            if phase not in phases:
                continue
            sweep_dir = fsdir / ('cifar_cnna_decay_untrained' if args.untrained else 'cifar_cnna_decay_sweep')
            out_file=smdir / f'fgs_decay_sweep_cifar_cnna_{phase}.pkl'
            if args.high_pass:
                sweep_dir = sweep_dir.parent / (str(sweep_dir.parts[-1]) + f'_highpass_{floating_string(args.high_pass)}')
                out_file = out_file.parent / (str(out_file.stem) + f'_highpass_{floating_string(args.high_pass)}.pkl')
            elif args.untrained:
                out_file = out_file.parent / (str(out_file.stem) + f'_untrained.pkl')
            sm = fgs_decay_sweep(dataloader=dl,
                arch=cnn_actually, sweep_dir = sweep_dir,
                plot=False, plot_dir=pa.Path('./plots/fgs_decay_sweep_cifar_cnna'), plot_suffix=phase,
                out_file=out_file,
                progress=args.progress) 
    elif args.sweep == 'decay_linear_cnn':
        cfdl = cifardl(batch_size=64, num_workers=8, half=False, 
            subsample=args.subsample, high_pass=args.high_pass)
        for phase, dl in cfdl.items():
            if phase not in phases:
                continue
            sweep_dir = fsdir / ('cifar_linear_cnn_decay_untrained' if args.untrained else 'cifar_linear_cnn_decay_sweep')
            out_file=smdir / f'fgs_decay_sweep_cifar_linear_cnn_{phase}.pkl'
            if args.high_pass:
                sweep_dir = sweep_dir.parent / (str(sweep_dir.parts[-1]) + f'_highpass_{floating_string(args.high_pass)}')
                out_file = out_file.parent / (str(out_file.stem) + f'_highpass_{floating_string(args.high_pass)}.pkl')
            elif args.untrained:
                out_file = out_file.parent / (str(out_file.stem) + f'_untrained.pkl')
            sm = fgs_decay_sweep(dataloader=dl,
                arch=cnn_actually, sweep_dir = sweep_dir,
                plot=False, 
                plot_dir=pa.Path('./plots/fgs_decay_sweep_cifar_linear_cnn'), 
                plot_suffix=phase,
                out_file=out_file,
                progress=args.progress,
                linear_hack=True) 
    elif args.sweep == 'decay_vgg':
        cfdl = cifardl(batch_size=64, num_workers=8, half=False, 
            subsample=args.subsample, high_pass=args.high_pass)
        for phase, dl in cfdl.items():
            if phase not in phases:
                continue
            sweep_dir = fsdir / ('cifar_vgg_decay_untrained' if args.untrained else 'cifar_vgg_decay_sweep')
            out_file=smdir / f'fgs_decay_sweep_cifar_vgg_{phase}.pkl'
            if args.high_pass:
                sweep_dir = sweep_dir.parent / (str(sweep_dir.parts[-1]) + f'_highpass_{floating_string(args.high_pass)}')
                out_file = out_file.parent / (str(out_file.stem) + f'_highpass_{floating_string(args.high_pass)}.pkl')
            elif args.untrained:
                out_file = out_file.parent / (str(out_file.stem) + f'_untrained.pkl')
            sm = fgs_decay_sweep(dataloader=dl,
                arch=vggc, sweep_dir = sweep_dir,
                plot=False, plot_dir=pa.Path('./plots/fgs_decay_sweep_cifar_cnna'), plot_suffix=phase,
                out_file=out_file,
                progress=args.progress)
    elif args.sweep == 'decay_imagenette_vgg':
        imdl = imagenettedl(batch_size=8, num_workers=4, high_pass=args.high_pass, half=False, subsample=args.subsample)
        for phase, dl in imdl.items():
            if phase not in phases:
                continue
            sweep_dir = fsdir / ('imagenette_vgg_decay_untrained' if args.untrained else 'imagenette_vgg_decay_sweep')
            out_file=smdir / f'fgs_decay_sweep_imagenette_vgg_{phase}.pkl'
            if args.high_pass:
                sweep_dir = sweep_dir.parent / (str(sweep_dir.parts[-1]) + f'_highpass_{floating_string(args.high_pass)}')
                out_file = out_file.parent / (str(out_file.stem) + f'_highpass_{floating_string(args.high_pass)}.pkl')
            elif args.untrained:
                out_file = out_file.parent / (str(out_file.stem) + f'_untrained.pkl')
            sm = fgs_decay_sweep(dataloader=dl,
                arch=imagenette_vgg, sweep_dir = sweep_dir,
                plot=False, plot_dir=pa.Path('./plots/fgs_decay_sweep_cifar_cnna'), plot_suffix=phase,
                out_file=out_file,
                dist_hack=True,
                progress=args.progress)
    elif args.sweep == 'depth_rnc':
        cfdl = cifardl(batch_size=64, num_workers=8, half=False, 
            subsample=args.subsample, high_pass=args.high_pass)
        for phase, dl in cfdl.items():
            if phase not in phases:
                continue
            sweep_dir = fsdir / ('cifar_resnet_depth_untrained' if args.untrained else 'cifar_resnet_depth_sweep')
            out_file=smdir / f'fgs_depth_sweep_cifar_rnc_{phase}.pkl'
            if args.high_pass:
                sweep_dir = sweep_dir.parent / (str(sweep_dir.parts[-1]) + f'_highpass_{floating_string(args.high_pass)}')
                out_file = out_file.parent / (str(out_file.stem) + f'_highpass_{floating_string(args.high_pass)}.pkl')
            elif args.untrained:
                out_file = out_file.parent / (str(out_file.stem) + f'_untrained.pkl')
            sm = fgs_depth_sweep(dataloader=dl,
                arch_family=rnc, sweep_dir = sweep_dir,
                plot=False, plot_dir=pa.Path('./plots/fgs_depth_sweep_cifar_rnc'), plot_suffix=phase,
                out_file=out_file,
                progress=args.progress)  
    elif args.sweep == 'depth_cnna':
        cfdl = cifardl(batch_size=64, num_workers=8, half=False, 
            subsample=args.subsample, high_pass=args.high_pass)
        for phase, dl in cfdl.items():
            if phase not in phases:
                continue
            sweep_dir = fsdir / ('cifar_cnna_depth_untrained' if args.untrained else 'cifar_cnna_depth_sweep')
            out_file=smdir / f'fgs_depth_sweep_cifar_cnna_{phase}.pkl'
            if args.high_pass:
                sweep_dir = sweep_dir.parent / (str(sweep_dir.parts[-1]) + f'_highpass_{floating_string(args.high_pass)}')
                out_file = out_file.parent / (str(out_file.stem) + f'_highpass_{floating_string(args.high_pass)}.pkl')
            elif args.untrained:
                out_file = out_file.parent / (str(out_file.stem) + f'_untrained.pkl')
            sm = fgs_depth_sweep(dataloader=dl,
                arch_family=cnn_actually, sweep_dir = sweep_dir,
                plot=False, plot_dir=pa.Path('./plots/fgs_depth_sweep_cifar_cnna'), plot_suffix=phase,
                out_file=out_file,
                progress=args.progress)  
    elif args.sweep == 'depth_linear_cnn':
        cfdl = cifardl(batch_size=64, num_workers=8, half=False, 
            subsample=args.subsample, high_pass=args.high_pass)
        for phase, dl in cfdl.items():
            if phase not in phases:
                continue
            sweep_dir = fsdir / ('cifar_linear_cnn_depth_untrained' if args.untrained else 'cifar_linear_cnn_depth_sweep')
            out_file=smdir / f'fgs_depth_sweep_cifar_linear_cnn_{phase}.pkl'
            if args.high_pass:
                sweep_dir = sweep_dir.parent / (str(sweep_dir.parts[-1]) + f'_highpass_{floating_string(args.high_pass)}')
                out_file = out_file.parent / (str(out_file.stem) + f'_highpass_{floating_string(args.high_pass)}.pkl')
            elif args.untrained:
                out_file = out_file.parent / (str(out_file.stem) + f'_untrained.pkl')
            sm = fgs_depth_sweep(dataloader=dl,
                arch_family=cnn_actually, sweep_dir = sweep_dir,
                plot=False, 
                plot_dir=pa.Path('./plots/fgs_depth_sweep_cifar_linear_cnn'), 
                plot_suffix=phase,
                out_file=out_file,
                progress=args.progress,
                linear_hack=True)  
    elif args.sweep == 'depth_vgg':
        cfdl = cifardl(batch_size=64, num_workers=8, half=False, 
            subsample=args.subsample, high_pass=args.high_pass)
        for phase, dl in cfdl.items():
            if phase not in phases:
                continue
            sweep_dir = fsdir / ('cifar_vgg_depth_untrained' if args.untrained else 'cifar_vgg_depth_sweep')
            out_file=smdir / f'fgs_depth_sweep_cifar_vgg_{phase}.pkl'
            if args.high_pass:
                sweep_dir = sweep_dir.parent / (str(sweep_dir.parts[-1]) + f'_highpass_{floating_string(args.high_pass)}')
                out_file = out_file.parent / (str(out_file.stem) + f'_highpass_{floating_string(args.high_pass)}.pkl')
            elif args.untrained:
                out_file = out_file.parent / (str(out_file.stem) + f'_untrained.pkl')
            sm = fgs_depth_sweep(dataloader=dl,
                arch_family=vggc, sweep_dir = sweep_dir,
                plot=False, plot_dir=pa.Path('./plots/fgs_depth_sweep_cifar_cnna'), plot_suffix=phase,
                out_file=out_file,
                progress=args.progress)
    elif args.sweep == 'depth_imagenette_vgg':
        imdl = imagenettedl(batch_size=8, num_workers=4, high_pass=args.high_pass, half=False, subsample=args.subsample)
        for phase, dl in imdl.items():
            if phase not in phases:
                continue
            sweep_dir = fsdir / ('imagenette_vgg_depth_untrained' if args.untrained else 'imagenette_vgg_depth_sweep')
            out_file=smdir / f'fgs_depth_sweep_imagenette_vgg_{phase}.pkl'
            if args.high_pass:
                sweep_dir = sweep_dir.parent / (str(sweep_dir.parts[-1]) + f'_highpass_{floating_string(args.high_pass)}')
                out_file = out_file.parent / (str(out_file.stem) + f'_highpass_{floating_string(args.high_pass)}.pkl')
            elif args.untrained:
                out_file = out_file.parent / (str(out_file.stem) + f'_untrained.pkl')
            sm = fgs_depth_sweep(dataloader=dl,
                arch_family=imagenette_vgg, sweep_dir = sweep_dir,
                plot=False, plot_dir=pa.Path('./plots/fgs_depth_sweep_cifar_cnna'), plot_suffix=phase,
                out_file=out_file,
                dist_hack=True,
                progress=args.progress)
    elif args.sweep == 'alpha_al':
        dldl = lwndl(batch_size=16, num_workers=8,
            subsample=args.subsample)
        for phase, dl in dldl.items():
            if phase not in phases:
                continue
            sweep_dir = results_dir / 'freq-sens/deadleaves_alexnet_alpha_sweep'
            out_file=smdir / f'fgs_alpha_sweep_lwn_al_{phase}.pkl'
            sm = fgs_decay_sweep(dataloader=dl,
                arch=al, sweep_dir = sweep_dir,
                plot=False, plot_dir=pa.Path('./plots/fgs_decay_sweep_cifar_cnna'), plot_suffix=phase,
                out_file=out_file,
                lwn_hack=True,
                progress=args.progress) 