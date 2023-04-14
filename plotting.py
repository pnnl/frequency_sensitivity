"""
Plotting functions.
"""

import torch as ch
from torch import fft 
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import pathlib as pa
import pickle as pkl
import numpy as np
snsrcd = {

    'font.family': 'serif',
}
sns.set_theme(context='paper',rc=snsrcd)

def shifty(x: ch.Tensor or dict or list or tuple):
    if isinstance(x, dict):
        return {k: shifty(v) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return [shifty(v) for v in x]
    else:
        return  fft.fftshift(x, dim=(-2,-1))
def npify(x: ch.Tensor or dict or list):
    if isinstance(x, dict):
        return {k: npify(v) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return [npify(v) for v in x]
    else:
        return  x.permute(1,2,0).numpy()
def zoomzoom(x: ch.Tensor or dict or list, 
    zoom: int):
    if isinstance(x, dict):
        return {k: zoomzoom(v, zoom) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return [zoomzoom(v, zoom) for v in x]
    else:
        return x[:, 
            x.shape[1]//2- x.shape[1]//(2*zoom):x.shape[1]//2+ x.shape[1]//(2*zoom),  
            x.shape[2]//2- x.shape[2]//(2*zoom):x.shape[2]//2+ x.shape[2]//(2*zoom)]
def leaves(x: ch.Tensor or dict or list):
    if isinstance(x, dict):
        ans = []
        for k, v in x.items():
            ans += leaves(v)
        return ans
    elif isinstance(x, (list, tuple)):
        ans = []
        for v in x:
            ans += leaves(v)
        return ans
    else:
        return [x]
def logparam_leaves(x: dict or list, add_one: bool=False):
    if isinstance(x, dict):
        return {k: logparam_leaves(v, add_one=add_one) for k, v in x.items()}
    else:
        if len(x) == 2:
            return np.log((1.0 if add_one else 0.0)+ x[0]), x[1]/((1.0 if add_one else 0.0)+x[0])
        else:
            return ([x[0]] + [np.log((1.0 if add_one else 0.0)+x[1]), x[2]/((1.0 if add_one else 0.0)+x[1])] + \
                ([x[3]] if len(x)>3 else []))
def divide_leaves(x: dict or list, denom: float or str = None, 
    distributions: bool=False):
    if isinstance(x, dict):
        return {k: divide_leaves(v, denom=denom, 
            distributions=distributions) for k, v in x.items()}
    else:
        if isinstance(denom, float):
            return [v/denom for v in x]
        elif denom == 'max':
            return [v/v.max() for v in x]
        elif denom == 'sum':
            if distributions:
                s = x[1].sum()
                return [(v/s if i in [1,2] else v) for i, v in enumerate(x)]
            else:
                return [v/v.sum() for v in x]
        elif denom == 'integral':
            if distributions:
                # midpoint rule of freshman calc
                integral = np.dot(
                    0.5*(x[1][1:] + x[1][:-1]),
                    np.diff(x[0])
                )
                return [(v/integral if i in [1,2] else v) for i, v in enumerate(x)]
            else:
                return [v/v.sum() for v in x]

def smooth_leaves(x: dict or list or tuple or ch.Tensor, smooth:int=1):
    if isinstance(x, dict):
        return {k: smooth_leaves(v,smooth=smooth) for k, v in x.items()}
    elif isinstance(x, tuple([list, tuple])):
        t = np.arange(-smooth, smooth+1)
        smther = np.ones(t.shape)
        smther /= smther.sum()
        ans = [np.convolve(v, smther, mode='valid') for v in x[:2]]
        if len(x) > 2:
            ans = ans + [np.sqrt(np.convolve(x[2]**2, smther, mode='valid'))]
        if len(x) > 3:
            ans = ans + [x[3]]
        return ans
    else:
        t = np.arange(-smooth, smooth+1)
        smther = np.ones(t.shape)
        smther /= smther.sum()
        return np.convolve(x, smther, mode='valid')

def radial_pushforward(x: dict or list[ch.Tensor], 
    ignore_std:bool=False,
    debug:bool=False,
    bin:bool=False) -> np.ndarray:
    if isinstance(x, ch.Tensor):
        x = [x]
        ignore_std = True
    if isinstance(x, dict):
        return {k: radial_pushforward(v,ignore_std=ignore_std,debug=debug) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        # npify
        x = [v.permute(1,2,0).numpy() for v in x]
        # gah sketchy
        if ignore_std:
            x = x[0]
        else:
            x, xstd = x 
        xrad, yrad = float(x.shape[0]/2), float(x.shape[1]/2)
        mgx, mgy = np.meshgrid(np.linspace(-xrad, xrad, x.shape[0]), 
            np.linspace(-yrad, yrad, x.shape[1]), indexing='xy')
        r = np.sqrt(mgx**2 + mgy**2)
        imr: np.ndarray = None
        imrcounts: np.ndarray = None
        if bin:
            imrcounts, bin_edges = np.histogram(r, bins='fd')
            imr = (bin_edges[:-1]+bin_edges[1:])/2
        else:
            imr = np.unique(r.flatten())
            imr = np.sort(imr, axis=0)
        if debug:
            tempfig, tempax = plt.subplots(1,1, figsize=(5,4))
            mask = (r==imr[-1])
            c = tempax.matshow(r*mask)
            plt.colorbar(c, ax= tempax)
        if ignore_std:
            ans = np.zeros_like(imr)
        else:
            ans, ansstd = np.zeros_like(imr), np.zeros_like(imr)
        if bin:
            for i in range(bin_edges.shape[0]-1):
                indicator1 = np.reshape((r>=bin_edges[i]), r.shape +(1,))
                indicator2 = np.reshape((r<=bin_edges[i+1]), r.shape +(1,))
                indicator = indicator1*indicator2
                ans[i] = np.sum(indicator*x)/np.sum(indicator)
                if not ignore_std:
                    ansstd[i] = np.sqrt(np.sum(indicator*xstd**2)/np.sum(indicator))
        else:
            imrcounts = np.zeros_like(imr)
            for i in range(imr.shape[0]):
                indicator = np.reshape((r==imr[i]), r.shape +(1,))
                imrcounts[i] = np.sum(indicator)
                ans[i] = np.sum(indicator*x)/imrcounts[i]
                if not ignore_std:
                    ansstd[i] = np.sqrt(np.sum(indicator*xstd**2)/imrcounts[i])
        if debug:
            return imr, ans, ansstd, imrcounts
        else:
            if ignore_std:
                return imr, ans
            else:
                return imr, ans, ansstd
def frequency_image(x: ch.Tensor or dict[ch.Tensor], 
    metric: str = 'meannorm',
    logparam:bool=False, lognrm:bool=False, 
    zoom:int=None, glob_max:bool=True,
    ticks: bool=True):
    if isinstance(x, dict):
        x = {k: v for (k, v) in sorted(list(x.items()), key= lambda x: float(x[0]))}
    x = shifty(x)
    if zoom:
        x = zoomzoom(x, zoom)
    if logparam:
        x = logparam_leaves(x, add_one=True)
    if glob_max:
        globmax = max(*leaves(x))
        x = divide_leaves(x, denom=globmax)
    else:
        x = divide_leaves(x, denom='max')
    x = npify(x)
    if isinstance(x, dict):
        fig, ax = plt.subplot_mosaic([list(x.keys())], figsize=(len(x.keys())*4,4))
        for k, v in x.items():
            if lognrm:
                im = ax[k].imshow(v[metric][0], norm=LogNorm())
                plt.colorbar(im, ax=ax[k])
            else:
                ax[k].imshow(v[metric][0])
            if ticks:
                # want about 8 ticks on each axis
                deltax = v[metric][0].shape[0]//8
                deltay = v[metric][0].shape[1]//8
                ax[k].set_xticks([i for i in range(v[metric][0].shape[0]) if i%deltax==0])
                ax[k].set_yticks([i for i in range(v[metric][0].shape[1])  if i%deltay==0])
                ax[k].set_xticklabels([i - v[metric][0].shape[0]//2 for i in range(v[metric][0].shape[0]) if i%deltax==0])
                ax[k].set_yticklabels([i - v[metric][0].shape[1]//2 for i in range(v[metric][0].shape[1])  if i%deltay==0])
            else: 
                ax[k].set_xticks([])
                ax[k].set_yticks([])
            if isinstance(k, int):
                ax[k].set_title(f'depth = {k}')
            else:
                ax[k].set_title(rf'$\lambda$ = {float(k):.3e}')
    else:
        fig, ax = plt.subplots(1,1, figsize=(4,4))
        if lognrm:
            im = ax.imshow(x, norm=LogNorm())
            plt.colorbar(im, ax=ax)
        else:
            ax.imshow(x)
        ax.set_xticks([i for i in range(x.shape[0]) if i%5==0])
        ax.set_yticks([i for i in range(x.shape[1])  if i%5==0])
        ax.set_xticklabels([i - x.shape[0]//2 for i in range(x.shape[0]) if i%5==0])
        ax.set_yticklabels([i - x.shape[1]//2 for i in range(x.shape[1])  if i%5==0])
    return fig, ax

def radial_freq_dist(x: ch.Tensor or dict[ch.Tensor], 
    metric = 'meannorm',
    logparam:bool=False, 
    distributions:bool=False, debug:bool = False,
    smooth:int = None, return_data: bool=False,
    error_bars:bool=True,
    input_data: bool = False,
    sortkeys:bool=True,
    ylabel_logscale:bool=False,
    xscale_log: bool=False,
    legend_override:str=None,
    include_legend: bool = False,
    share:bool=False):
    if not input_data:
        # if x is a dict w/ numeric keys, ensure it is sorted:
        if isinstance(x, dict) and sortkeys:
            x = {k: v for (k, v) in sorted(list(x.items()), key= lambda x: float(x[0]))}
        x = shifty(x)
        positivity_test = all([ch.all(v >= 0) for  v in leaves(x)])
        assert positivity_test, f'need positive values and positivity test = {positivity_test}!'
        if debug:
            if isinstance(x, dict):
                test = leaves(x)[0]
                imr, fs, fsstd, imrcounts = radial_pushforward(test, debug=True)
                tempfig, tempax = plt.subplots(1,1, figsize=(8,4))
                tempax.bar(imr, imrcounts)
                tempax.set_xlabel('radius')
                tempax.set_ylabel('samples')
        x = radial_pushforward(x)
    else:
        if (share and sortkeys):
            x = {l: {k: v for (k, v) in sorted(list(w.items()), key= lambda x: float(x[0]))} for l, w in x.items()}
        elif (isinstance(x, dict) and sortkeys):
            x = {k: v for (k, v) in sorted(list(x.items()), key= lambda x: float(x[0]))}
    if distributions:
        x = divide_leaves(x, denom='integral', distributions=True)
    if smooth:
        x = smooth_leaves(x, smooth=smooth)
    if logparam:
        x = logparam_leaves(x)
    def paint_ax(x, ax):
        if isinstance(x, dict):
            def labeller(k):
                if isinstance(k, int):
                    return f'depth = {k}'
                elif isinstance(k, float):
                    if legend_override:
                        return f'{legend_override} = {float(k):.2e}'
                    else:
                        return f'decay = {float(k):.2e}'
                else:
                    return k 
            cmap = plt.get_cmap('viridis')
            num_clrs = len(x.keys())
            clrs = [cmap(1.0*i/num_clrs) for i in range(num_clrs)]
            for i, (k, v) in enumerate(x.items()):
                if error_bars:
                    ax.plot(v[metric][0], v[metric][1],
                        label=labeller(k), color=clrs[i])
                    ax.fill_between(v[metric][0], v[metric][1]-v[metric][2],
                        v[metric][1]+v[metric][2], color=clrs[i],
                        alpha=0.2)
                else:
                    ax.plot(v[metric][0], v[metric][1], label=labeller(k), color=clrs[i])
            if include_legend:
                ax.legend()
        else:
            if error_bars:
                ax.plot(v[metric][0], v[metric][1])
                ax.fill_between(v[metric][0], v[metric][1]-v[metric][2],
                    v[metric][1]+v[metric][2], alpha=0.2)
            else:
                ax.plot(x[0], x[1])
        ax.set_xlabel((f'frequency radius' if logparam else f'frequency radius'))
        ax.set_ylabel(
            (f'log gradient sensitivity' if (logparam or ylabel_logscale)
                else f'gradient sensitivity')
        )
        if xscale_log:
            ax.set_xscale('log')
    if share:
        fig, ax = plt.subplots(1, 
            len(list(x.items())), 
            figsize=(8*len(list(x.items())),4),)
        if len(list(x.items())) == 1:
            for i, (k, v) in enumerate(x.items()):
                paint_ax(x=v, ax=ax)
                ax.set_title(k)    
        else:
            for i, (k, v) in enumerate(x.items()):
                paint_ax(x=v, ax=ax[i])
                ax[i].set_title(k)
    else:
        fig, ax = plt.subplots(1,1, figsize=(8,4))
        paint_ax(x=x, ax=ax)
    if debug:
        return fig, ax, imrcounts
    elif return_data:
        return fig, ax, x
    else:
        return fig, ax
