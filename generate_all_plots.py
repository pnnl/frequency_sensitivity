"""
TODO: add linear CNNs
"""

from freq_sens import *
import pandas as pd
import re
import itertools as it
from plotting import *
from datasets import *
from torchvision.transforms import ToPILImage
topil = ToPILImage()
import pandas as pd
import argparse as ap
device = ch.device('cuda:0')

pp_dir = pa.Path('./paper_plots/')

# Utilities

def sm_dict(smdir: str or pa.Path):
    ans = {}
    for f in smdir.iterdir():
        with open(f, 'rb') as g:
            ans[f.stem] = pkl.load(g)
    return ans

def sm_averager(sm: dict[str, dict], metrics: list[str] = ['meannorm', 'rms']):
    new_sm = {}
    for k in sm.keys():
        new_sm[k] = {}
        for m in metrics:
            v = ch.stack([sm[k][l][m][0] for l in sm[k].keys()], dim=0)
            new_sm[k][m] = v.mean(dim=0), v.var(dim=0).sqrt()
    return new_sm

def rp_averager(sm: dict[str, dict], metrics: list[str] = ['meannorm', 'rms'],
    smooth:int=None, distributions:bool=False, logparam: bool=False,
    no_shift: bool=False, frequency_cutoff:int=None,
    bin:bool=False):
    if not no_shift:
        shifted_sm =shifty(sm)
    else:
        shifted_sm = sm
    new_sm = {}
    for k in sm.keys():
        new_sm[k] = {}
        for m in metrics:
            rps =[radial_pushforward(shifted_sm[k][l][m], bin=bin) for l in shifted_sm[k].keys()]
            if frequency_cutoff:
                rps = [[y[:frequency_cutoff] for y in x] for x in rps]
            if distributions:
                rps = [divide_leaves(x, denom='sum', distributions=True) for x in rps]
            if smooth != None:
                rps = [smooth_leaves(x, smooth=smooth) for x in rps]
            if logparam:
                rps = [logparam_leaves(x) for x in rps]
            imr = np.mean(np.array([imr for imr, ans, ansstd in rps]), axis=0)
            ans = np.mean(np.array([ans for imr, ans, ansstd in rps]), axis=0)
            ansstd = np.sqrt(np.var(np.array([ans for imr, ans, ansstd in rps]), axis=0))
            new_sm[k][m] = (imr, ans, ansstd)
    return new_sm

def mom_averager(sm: dict[str, dict], metrics: list[str] = ['meannorm', 'rms'],
    smooth:int=None, moment:int=1):
    shifted_sm =shifty(sm)
    new_sm = {}
    for k in sm.keys():
        new_sm[k] = {}
        for m in metrics:
            rps =[radial_pushforward(shifted_sm[k][l][m]) for l in shifted_sm[k].keys()]
            if smooth != None:
                rps = [smooth_leaves(x, smooth=smooth) for x in rps]
            moms = [moments(x[0], x[1], moment=moment) for x in rps]
            ans = np.mean(np.array(moms))
            ansstd = np.sqrt(np.var(np.array(moms)))
            new_sm[k][m] = (ans, ansstd)
    return new_sm

def decay_sweep_acc_dict(sweep_dir:str or pa.Path, sm_dict: dict[str, dict] = None, legacy:bool=False):
    ans = {}
    for d in sweep_dir.iterdir():
        ans[d.stem] = {}
        for e in d.iterdir():
            if legacy:
                with open(e / 'log.txt', 'r') as f:
                    log = f.read()
                if not re.search("'model': RNC", log):
                    continue
            vc = ch.load(e/'val_curve_acc.pt')
            l = floating_string(d.stem)
            if sm_dict:
                if l in list(it.chain(*[[float(k) for k in v.keys()] for v in sm_dict.values()])):
                    ans[d.stem][e.stem] = vc.max().item()
            else:
                ans[d.stem][e.stem] = vc.max().item()
    return ans

def depth_sweep_acc_dict(sweep_dir:str or pa.Path, sm_dict: dict[dict] = None):
    ans = {}
    for d in sweep_dir.iterdir():
        ans[d.stem] = {}
        for e in d.iterdir():
            vc = ch.load(e/'val_curve_acc.pt')
            l = int(d.stem)
            if sm_dict:
                if l in list(it.chain(*[[float(k) for k in v.keys()] for v in sm_dict.values()])):
                    ans[d.stem][e.stem] = vc.max().item()
            else:
                ans[d.stem][e.stem] = vc.max().item()
    return ans

def moments(x: np.ndarray, p: np.ndarray, q: np.ndarray, moment: int = 1):
    if moment == 1:
        return (x*q).sum()/q.sum()
    else:
        return ((x**2)*q).sum()/q.sum()

def kldiv(x: np.ndarray, p: np.ndarray, q: np.ndarray):
    # make sure p, q are dists:
    p /= p.sum()
    q /= q.sum()
    return (p*np.log(p/q)).mean()

def totvar(x: np.ndarray, p: np.ndarray, q: np.ndarray):
    # make sure p, q are dists:
    p /= p.sum()
    q /= q.sum()
    return (1/2)*np.linalg.norm(p-q, ord=1)

def cossim(x: np.ndarray, p: np.ndarray, q: np.ndarray):
    return np.sum(p*q)/(np.linalg.norm(p)*np.linalg.norm(q))

# TODO: add a kldiv with the true image stats variant of mom_averager.
def div_averager(sm: dict[str, dict], 
    image_stats: ch.Tensor or np.ndarray,
    metrics: list[str] = ['meannorm', 'rms'],
    smooth:int=None,
    div: Callable = kldiv):
    shifted_sm =shifty(sm)
    new_sm = {}
    for k in sm.keys():
        new_sm[k] = {}
        for m in metrics:
            rps =[radial_pushforward(shifted_sm[k][l][m]) for l in shifted_sm[k].keys()]
            if smooth != None:
                rps = [smooth_leaves(x, smooth=smooth) for x in rps]
            # order matters here (for KL). 
            divs = [div(x[0], image_stats, x[1],) for x in rps]
            ans = np.mean(np.array(divs))
            ansstd = np.sqrt(np.var(np.array(divs)))
            new_sm[k][m] = (ans, ansstd)
    return new_sm

if __name__=='__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('--pp_dir', type=str, default=None)
    parser.add_argument('--unnormalized', action='store_true')
    args = parser.parse_args()
    if args.pp_dir:
        pp_dir = pa.Path(args.pp_dir)
    # Dataset freq stats

    covdir = pa.Path('./results/freq-sens/covariances')
    dldict = {
        'cifar': cifardl, 
        'imagenette': imagenettedl
    }
    alphas = [0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    for a in alphas:
        dldict[f'lwn_{floating_string(a)}'] = (lwndl, a)

    pmaps = [covdir/ f'{k}_train_power.pt' for k in dldict.keys() if not re.search('lwn', k)]
    fig, a = plt.subplots(1, len(pmaps), figsize=(3*len(pmaps), 3))
    for i, f in enumerate(pmaps):
        # print(f)
        x = shifty(ch.load(str(f)))
        # if re.search('cifar', f.stem):
        #     x = zoomzoom(x, zoom=2)
        # elif re.search('imagenette', f.stem):
        #     x = zoomzoom(x, zoom=2*int(224/32))
        # elif re.search('lwn', f.stem):
        #     x = zoomzoom(x, zoom=2*int(64/32))
        # optionally, use log_scale (add 1 to ensure positivity)
        x =  (1.0* x).log()
        x -= x.min()
        x /= x.max()
        a[i].imshow(topil(x))
        a[i].set_title(' '.join(f.stem.split('_')[:-2]).replace('lwn', 'wmm'))
        a[i].set_xticks([int(i) for i in np.linspace(0,x.shape[-1], 5)])
        a[i].set_yticks([int(i) for i in np.linspace(0,x.shape[-2], 5)])
        a[i].set_xticklabels([int(i) for i in np.linspace(-x.shape[-1]//2,x.shape[-1]//2, 5)])
        a[i].set_yticklabels([int(i) for i in np.linspace(-x.shape[-2]//2,x.shape[-2]//2, 5)])
    fig.tight_layout()
    fig.savefig(pp_dir/'pmaps_natural.png', dpi=200)
    plt.close(fig)

    pmaps = [covdir/ f'{k}_train_power.pt' for k in dldict.keys()]
    fig, a = plt.subplots(1, len(pmaps), figsize=(3*len(pmaps), 3))
    for i, f in enumerate(pmaps):
        # print(f)
        x = shifty(ch.load(str(f)))
        # if re.search('cifar', f.stem):
        #     x = zoomzoom(x, zoom=2)
        # elif re.search('imagenette', f.stem):
        #     x = zoomzoom(x, zoom=2*int(224/32))
        # elif re.search('lwn', f.stem):
        #     x = zoomzoom(x, zoom=2*int(64/32))
        # optionally, use log_scale (add 1 to ensure positivity)
        x =  (1.0* x).log()
        x -= x.min()
        x /= x.max()
        a[i].imshow(topil(x))
        a[i].set_title(' '.join(f.stem.split('_')[:-2]).replace('lwn', 'wmm'))
        a[i].set_xticks([int(i) for i in np.linspace(0,x.shape[-1], 5)])
        a[i].set_yticks([int(i) for i in np.linspace(0,x.shape[-2], 5)])
        a[i].set_xticklabels([int(i) for i in np.linspace(-x.shape[-1]//2,x.shape[-1]//2, 5)])
        a[i].set_yticklabels([int(i) for i in np.linspace(-x.shape[-2]//2,x.shape[-2]//2, 5)])
    fig.tight_layout()
    fig.savefig(pp_dir/'pmaps.png', dpi=200)
    plt.close(fig)

    pmaps = {k: covdir/ f'{k}_train_power.pt' for k in dldict.keys()}
    for k, v in pmaps.items():
        x = ch.load(str(v))
        x = shifty(x)
        x = radial_pushforward(x, 
            # bin=True
            )
        smoothing_radius = 3
        if re.search('image', k):
            smoothing_radius=5
        x = smooth_leaves(x, smooth=smoothing_radius)
        pmaps[k] = x

    fig, a = plt.subplots(1, len(pmaps), figsize=(3*len(pmaps), 3))
    for i, (k, x) in enumerate(pmaps.items()):
        a[i].plot(x[0], x[1])
        a[i].set_xlabel('frequency magnitude')
        a[i].set_ylabel('variance')
        a[i].set_title(k)
        # a[i].set_xlim(0, np.sqrt(2)*8)
        a[i].set_yscale('log')
        a[i].set_xscale('log')
    fig.tight_layout()
    fig.savefig(pp_dir / 'pmaps_radial_loglog.png', dpi=200)
    plt.close(fig)

    for i, (k, x) in enumerate(pmaps.items()):
        if re.search('lwn', k):
            continue
        else:
            fig, a = plt.subplots(1, 1, figsize=(8, 4))
            a.plot(x[0], x[1])
            a.set_xlabel('frequency magnitude')
            a.set_ylabel('variance')
            a.set_title(k)
            # a.set_xlim(0, np.sqrt(2)*8)
            a.set_yscale('log')
            a.set_xscale('log')
            fig.tight_layout()
            fig.savefig(pp_dir / f'pmaps_radial_loglog_{k}.png', dpi=200)
            plt.close(fig)

    pmaps_hp = [covdir/ f'{k}_train_power_hp.pt' for k in dldict.keys() if not re.search('lwn', k)]
    fig, a = plt.subplots(1, len(pmaps_hp), figsize=(3*len(pmaps_hp), 3))
    for i, f in enumerate(pmaps_hp):
        # print(f)
        x = shifty(ch.load(str(f)))
        # if re.search('cifar', f.stem):
        #     x = zoomzoom(x, zoom=2)
        # elif re.search('imagenette', f.stem):
        #     x = zoomzoom(x, zoom=2*int(224/32))
        # optionally, use log_scale (add 1 to ensure positivity)
        x =  (1.0* x).log()
        x -= x.min()
        x /= x.max()
        a[i].imshow(topil(x))
        a[i].set_title(' '.join(f.stem.split('_')[:-2]).replace('lwn', 'wmm'))
        a[i].set_xticks([int(i) for i in np.linspace(0,x.shape[-1], 5)])
        a[i].set_yticks([int(i) for i in np.linspace(0,x.shape[-2], 5)])
        a[i].set_xticklabels([int(i) for i in np.linspace(-x.shape[-1]//2,x.shape[-1]//2, 5)])
        a[i].set_yticklabels([int(i) for i in np.linspace(-x.shape[-2]//2,x.shape[-2]//2, 5)])
    fig.tight_layout()
    fig.savefig(pp_dir/'pmaps_hp.png', dpi=200)
    plt.close(fig)

    pmaps_hp = {k: covdir/ f'{k}_train_power_hp.pt' for k in dldict.keys() if not re.search('lwn', k)}
    for k, v in pmaps_hp.items():
        x = ch.load(str(v))
        x = shifty(x)
        x = radial_pushforward(x, 
            # bin=True,
            )
        smoothing_radius = 3
        if re.search('image', k):
            smoothing_radius=5
        x = smooth_leaves(x, smooth=smoothing_radius)
        pmaps_hp[k] = x

    fig, a = plt.subplots(1, len(pmaps_hp), figsize=(3*len(pmaps_hp), 3))
    for i, (k, x) in enumerate(pmaps_hp.items()):
        a[i].plot(x[0], x[1])
        a[i].set_xlabel('frequency magnitude')
        a[i].set_ylabel('variance')
        a[i].set_title(k)
        # a[i].set_xlim(0, np.sqrt(2)*8)
        a[i].set_yscale('log')
        a[i].set_xscale('log')
    fig.tight_layout()
    fig.savefig(pp_dir / 'pmaps_hp_radial_loglog.png', dpi=200)
    plt.close(fig)

    for i, (k, x) in enumerate(pmaps_hp.items()):
        fig, a = plt.subplots(1, 1, figsize=(8, 4))
        a.plot(x[0], x[1])
        a.set_xlabel('frequency magnitude')
        a.set_ylabel('variance')
        a.set_title(k)
        # a.set_xlim(0, np.sqrt(2)*8)
        a.set_yscale('log')
        a.set_xscale('log')
        fig.tight_layout()
        fig.savefig(pp_dir / f'pmaps_hp_radial_loglog_{k}.png', dpi=200)
        plt.close(fig)

    # Model sensitivity maps

    sm = sm_dict(smdir)

    pmihc = ch.load(covdir / 'imagenette_train_power_hp_hi_cutoff.pt')
    pmihc = shifty(pmihc)
    pmihc = radial_pushforward(pmihc)
    pmihc = smooth_leaves(pmihc, smooth=5)
    pmihc = pmihc
    fig, a = plt.subplots(1, 1, figsize=(8, 4))
    a.plot(pmihc[0], pmihc[1])
    a.set_xlabel('frequency magnitude')
    a.set_ylabel('variance')
    a.set_title('imagenette, higher threshold')
    # a.set_xlim(0, np.sqrt(2)*8)
    a.set_yscale('log')
    a.set_xscale('log')
    fig.tight_layout()
    fig.savefig(pp_dir / f'pmaps_hp_radial_loglog_imagenette_hicut.png', dpi=200)
    plt.close(fig)
    test_sm_args = {
        'decay': {
            'resnet': {'natural': {
                    'acc_dir': fsdir / 'cifar_resnet_decay_sweep',
                    'sm_key': 'fgs_decay_sweep_cifar_rnc_test',
                    'image_stats': pmaps['cifar'],
                },
                'high_pass_filtered': {
                    'acc_dir': fsdir / 'cifar_resnet_decay_sweep_highpass_5pt000e-01',
                    'sm_key': 'fgs_decay_sweep_cifar_rnc_train_highpass_5pt000e-01',
                    'image_stats': pmaps['cifar'],
                }},
            'mcnn': {'natural': {
                    'acc_dir': fsdir / 'cifar_mcnn_decay_sweep',
                    'sm_key': 'fgs_decay_sweep_cifar_mcnn_test',
                    'image_stats': pmaps['cifar'],
                },
                'high_pass_filtered': {
                    'acc_dir': fsdir / 'cifar_mcnn_decay_sweep_highpass_5pt000e-01',
                    'sm_key': 'fgs_decay_sweep_cifar_mcnn_test_highpass_5pt000e-01',
                    'image_stats': pmaps_hp['cifar'],
                }},
            'vggc': {'natural': {
                    'acc_dir': fsdir / 'cifar_vgg_decay_sweep',
                    'sm_key': 'fgs_decay_sweep_cifar_vgg_test',
                    'image_stats': pmaps['cifar'],
                },
                'high_pass_filtered': {
                    'acc_dir': fsdir / 'cifar_vgg_decay_sweep_highpass_5pt000e-01',
                    # clean: dropped top \lambda where training failed.
                    'sm_key': 'fgs_decay_sweep_cifar_vgg_test_highpass_5pt000e-01_CLEAN',
                    'image_stats': pmaps['cifar'],
                }},
            'vggi': {'natural': {
                    'acc_dir': fsdir / 'imagenette_vgg_decay_sweep',
                    'sm_key': 'fgs_decay_sweep_imagenette_vgg_test',
                    'image_stats': pmaps['imagenette'],
                },
                'high_pass_filtered': {
                    'acc_dir': fsdir / 'imagenette_vgg_decay_sweep_highpass_7pt140e-02',
                    # TODO: update
                    # 'sm_key': 'fgs_decay_sweep_imagenette_vgg_test_highpass_5pt000e-01',
                    'sm_key': 'fgs_decay_sweep_imagenette_vgg_test_highpass_7pt140e-02',
                    'image_stats': pmaps_hp['imagenette'],
                }},
            'vggi_hi_cutoff': {'natural': {
                    'acc_dir': fsdir / 'imagenette_vgg_decay_sweep',
                    'sm_key': 'fgs_decay_sweep_imagenette_vgg_test',
                    'image_stats': pmaps['imagenette'],
                },
                'high_pass_filtered': {
                    'acc_dir': fsdir / 'imagenette_vgg_decay_sweep_highpass_5pt000e-01',
                    # TODO: update
                    'sm_key': 'fgs_decay_sweep_imagenette_vgg_test_highpass_5pt000e-01',
                    # 'sm_key': 'fgs_decay_sweep_imagenette_vgg_test_highpass_7pt140e-02',
                    'image_stats': pmihc,
                }},
            'cnna': {'natural': {
                    'acc_dir': fsdir / 'cifar_cnna_decay_sweep',
                    'sm_key': 'fgs_decay_sweep_cifar_cnna_test',
                    'image_stats': pmaps['cifar'],
                },
                'high_pass_filtered': {
                    'acc_dir': fsdir / 'cifar_cnna_decay_sweep_highpass_5pt000e-01',
                    'sm_key': 'fgs_decay_sweep_cifar_cnna_test_highpass_5pt000e-01',
                    'image_stats': pmaps['cifar'],
                }},
            'linear_cnn': {'natural': {
                    'acc_dir': fsdir / 'cifar_linear_cnn_decay_sweep',
                    'sm_key': 'fgs_decay_sweep_cifar_linear_cnn_test',
                    'image_stats': pmaps['cifar'],
                },
                'high_pass_filtered': {
                    'acc_dir': fsdir / 'cifar_linear_cnn_decay_sweep_highpass_5pt000e-01',
                    'sm_key': 'fgs_decay_sweep_cifar_linear_cnn_test_highpass_5pt000e-01',
                    'image_stats': pmaps['cifar'],
                }},
        },
        'depth': {
            'resnet': {'natural': {
                    'acc_dir': fsdir / 'cifar_resnet_depth_sweep',
                    'sm_key': 'fgs_depth_sweep_cifar_rnc_test',
                    'image_stats': pmaps['cifar'],
                },
                'high_pass_filtered': {
                    'acc_dir': fsdir / 'cifar_resnet_depth_sweep_highpass_5pt000e-01',
                    'sm_key': 'fgs_depth_sweep_cifar_rnc_test_highpass_5pt000e-01',
                    'image_stats': pmaps['cifar'],
                }},
            'vggc': {'natural': {
                'acc_dir': fsdir / 'cifar_vgg_depth_sweep',
                'sm_key': 'fgs_depth_sweep_cifar_vgg_test',
                'image_stats': pmaps['cifar'],
            },
            'high_pass_filtered': {
                'acc_dir': fsdir / 'cifar_vgg_depth_sweep_highpass_5pt000e-01',
                'sm_key': 'fgs_depth_sweep_cifar_vgg_test_highpass_5pt000e-01',
                'image_stats': pmaps['cifar'],
            }},
            'vggi': {'natural':{
                'acc_dir': fsdir / 'imagenette_vgg_depth_sweep',
                'sm_key': 'fgs_depth_sweep_imagenette_vgg_test',
                'image_stats': pmaps['imagenette'],
            },
            'high_pass_filtered': {
                'acc_dir': fsdir / 'imagenette_vgg_depth_sweep_highpass_7pt140e-02',
                # TODO: update
                # 'sm_key': 'fgs_depth_sweep_imagenette_vgg_test_highpass_5pt000e-01',
                'sm_key': 'fgs_depth_sweep_imagenette_vgg_test_highpass_7pt140e-02',
                'image_stats': pmaps_hp['imagenette'],
            }},
            'vggi_hi_cutoff': {'natural':{
                'acc_dir': fsdir / 'imagenette_vgg_depth_sweep',
                'sm_key': 'fgs_depth_sweep_imagenette_vgg_test',
                'image_stats': pmaps['imagenette'],
            },
            'high_pass_filtered': {
                'acc_dir': fsdir / 'imagenette_vgg_depth_sweep_highpass_5pt000e-01',
                # TODO: update
                'sm_key': 'fgs_depth_sweep_imagenette_vgg_test_highpass_5pt000e-01',
                # 'sm_key': 'fgs_depth_sweep_imagenette_vgg_test_highpass_7pt140e-02',
                'image_stats': pmihc,
            }},
            'cnna': {'natural':{
                'acc_dir': fsdir / 'cifar_cnna_depth_sweep',
                'sm_key': 'fgs_depth_sweep_cifar_cnna_test',
                'image_stats': pmaps['cifar'],
            },
            'high_pass_filtered': {
                'acc_dir': fsdir / 'cifar_cnna_depth_sweep_highpass_5pt000e-01',
                'sm_key': 'fgs_depth_sweep_cifar_cnna_test_highpass_5pt000e-01',
                'image_stats': pmaps['cifar'],
            }},
            'linear_cnn': {'natural':{
                'acc_dir': fsdir / 'cifar_linear_cnn_depth_sweep',
                'sm_key': 'fgs_depth_sweep_cifar_linear_cnn_test',
                'image_stats': pmaps['cifar'],
            },
            'high_pass_filtered': {
                'acc_dir': fsdir / 'cifar_linear_cnn_depth_sweep_highpass_5pt000e-01',
                'sm_key': 'fgs_depth_sweep_cifar_linear_cnn_test_highpass_5pt000e-01',
                'image_stats': pmaps['cifar'],
            }},
        }
    }

    train_sm_args = {
        'decay': {
            'resnet': {'natural':{
                'acc_dir': fsdir / 'cifar_resnet_decay_sweep',
                'sm_key': 'fgs_decay_sweep_cifar_rnc_train',
                'image_stats': pmaps['cifar'],
            },
            'high_pass_filtered': {
                'acc_dir': fsdir / 'cifar_resnet_decay_sweep_highpass_5pt000e-01',
                'sm_key': 'fgs_decay_sweep_cifar_rnc_train_highpass_5pt000e-01',
                'image_stats': pmaps['cifar'],
            }},
            'mcnn': {'natural':{
                'acc_dir': fsdir / 'cifar_mcnn_decay_sweep',
                'sm_key': 'fgs_decay_sweep_cifar_mcnn_train',
                'image_stats': pmaps['cifar'],
            },
            'high_pass_filtered': {
                'acc_dir': fsdir / 'cifar_mcnn_decay_sweep_highpass_5pt000e-01',
                'sm_key': 'fgs_decay_sweep_cifar_mcnn_train_highpass_5pt000e-01',
                'image_stats': pmaps_hp['cifar'],
            }},
            'vggc': {'natural':{
                'acc_dir': fsdir / 'cifar_vgg_decay_sweep',
                'sm_key': 'fgs_decay_sweep_cifar_vgg_train',
                'image_stats': pmaps['cifar'],
            },
            'high_pass_filtered': {
                'acc_dir': fsdir / 'cifar_vgg_decay_sweep_highpass_5pt000e-01',
                'sm_key': 'fgs_decay_sweep_cifar_vgg_train_highpass_5pt000e-01',
                'image_stats': pmaps['cifar'],
            }},
            'vggi': {'natural':{
                'acc_dir': fsdir / 'imagenette_vgg_decay_sweep',
                'sm_key': 'fgs_decay_sweep_imagenette_vgg_train',
                'image_stats': pmaps['imagenette'],
            },
            'high_pass_filtered': {
                'acc_dir': fsdir / 'imagenette_vgg_decay_sweep_highpass_7pt140e-02',
                'sm_key': 'fgs_decay_sweep_imagenette_vgg_train_highpass_7pt140e-02',
                'image_stats': pmaps_hp['imagenette'],
            }},
            'cnna': {'natural':{
                'acc_dir': fsdir / 'cifar_cnna_decay_sweep',
                'sm_key': 'fgs_decay_sweep_cifar_cnna_train',
                'image_stats': pmaps['cifar'],
            },
            'high_pass_filtered': {
                'acc_dir': fsdir / 'cifar_cnna_decay_sweep_highpass_5pt000e-01',
                'sm_key': 'fgs_decay_sweep_cifar_cnna_train_highpass_5pt000e-01',
                'image_stats': pmaps['cifar'],
            }},
            'linear_cnn': {'natural':{
                'acc_dir': fsdir / 'cifar_linear_cnn_decay_sweep',
                'sm_key': 'fgs_decay_sweep_cifar_linear_cnn_train',
                'image_stats': pmaps['cifar'],
            },
            'high_pass_filtered': {
                'acc_dir': fsdir / 'cifar_linear_cnn_decay_sweep_highpass_5pt000e-01',
                'sm_key': 'fgs_decay_sweep_cifar_linear_cnn_train_highpass_5pt000e-01',
                'image_stats': pmaps['cifar'],
            }},
        },
        'depth': {
            'resnet': {'natural':{
                'acc_dir': fsdir / 'cifar_resnet_depth_sweep',
                'sm_key': 'fgs_depth_sweep_cifar_rnc_train',
                'image_stats': pmaps['cifar'],
            },
            'high_pass_filtered': {
                'acc_dir': fsdir / 'cifar_resnet_depth_sweep_highpass_5pt000e-01',
                'sm_key': 'fgs_depth_sweep_cifar_rnc_train_highpass_5pt000e-01',
                'image_stats': pmaps['cifar'],
            }},
            'vggc': {'natural':{
                'acc_dir': fsdir / 'cifar_vgg_depth_sweep',
                'sm_key': 'fgs_depth_sweep_cifar_vgg_train',
                'image_stats': pmaps['cifar'],
            },
            'high_pass_filtered': {
                'acc_dir': fsdir / 'cifar_vgg_depth_sweep_highpass_5pt000e-01',
                'sm_key': 'fgs_depth_sweep_cifar_vgg_train_highpass_5pt000e-01',
                'image_stats': pmaps['cifar'],
            }},
            'vggi': {'natural':{
                'acc_dir': fsdir / 'imagenette_vgg_depth_sweep',
                'sm_key': 'fgs_depth_sweep_imagenette_vgg_train',
                'image_stats': pmaps['imagenette'],
            },
            'high_pass_filtered': {
                'acc_dir': fsdir / 'imagenette_vgg_depth_sweep_highpass_7pt140e-02',
                'sm_key': 'fgs_depth_sweep_imagenette_vgg_train_highpass_7pt140e-02',
                'image_stats': pmaps_hp['imagenette'],
            }},
            'cnna': {'natural':{
                'acc_dir': fsdir / 'cifar_cnna_depth_sweep',
                'sm_key': 'fgs_depth_sweep_cifar_cnna_train',
                'image_stats': pmaps['cifar'],
            },
            'high_pass_filtered': {
                'acc_dir': fsdir / 'cifar_cnna_depth_sweep_highpass_5pt000e-01',
                'sm_key': 'fgs_depth_sweep_cifar_cnna_train_highpass_5pt000e-01',
                'image_stats': pmaps['cifar'],
            }},
            'linear_cnn': {'natural':{
                'acc_dir': fsdir / 'cifar_linear_cnn_depth_sweep',
                'sm_key': 'fgs_depth_sweep_cifar_linear_cnn_train',
                'image_stats': pmaps['cifar'],
            },
            'high_pass_filtered': {
                'acc_dir': fsdir / 'cifar_linear_cnn_depth_sweep_highpass_5pt000e-01',
                'sm_key': 'fgs_depth_sweep_cifar_linear_cnn_train_highpass_5pt000e-01',
                'image_stats': pmaps['cifar'],
            }},
        }
    }

    untrained_sm_args = {
        'decay': {
            'mcnn': {'natural': {
                    'acc_dir': fsdir / 'cifar_mcnn_decay_untrained',
                    'sm_key': 'fgs_decay_sweep_cifar_mcnn_test_untrained',
                    'image_stats': pmaps['cifar'],
                },},
        },
        'depth': {
            'resnet': {'natural': {
                    'acc_dir': fsdir / 'cifar_resnet_depth_untrained',
                    'sm_key': 'fgs_depth_sweep_cifar_rnc_test_untrained',
                    'image_stats': pmaps['cifar'],
                },},
            'vggc': {'natural': {
                'acc_dir': fsdir / 'cifar_vgg_depth_untrained',
                'sm_key': 'fgs_depth_sweep_cifar_vgg_test_untrained',
                'image_stats': pmaps['cifar'],
            },},
            'vggi': {'natural':{
                'acc_dir': fsdir / 'imagenette_vgg_depth_untrained',
                'sm_key': 'fgs_depth_sweep_imagenette_vgg_test_untrained',
                'image_stats': pmaps['imagenette'],
            },},
            'vggi_hi_cutoff': {'natural':{
                'acc_dir': fsdir / 'imagenette_vgg_depth_untrained',
                'sm_key': 'fgs_depth_sweep_imagenette_vgg_test_untrained',
                'image_stats': pmaps['imagenette'],
            },},
            'cnna': {'natural':{
                'acc_dir': fsdir / 'cifar_cnna_depth_untrained',
                'sm_key': 'fgs_depth_sweep_cifar_cnna_test_untrained',
                'image_stats': pmaps['cifar'],
            },},
            'linear_cnn': {'natural':{
                'acc_dir': fsdir / 'cifar_linear_cnn_depth_untrained',
                'sm_key': 'fgs_depth_sweep_cifar_linear_cnn_test_untrained',
                'image_stats': pmaps['cifar'],
            },},
        }
    }

    div_args = {
        'KL divergence': kldiv,
        'total variation': totvar,
        'cosine similarity': cossim,
        'first moment': moments,
    }

    def smplots(sm_args: dict[str, dict], div_args: dict[str, Callable],
        suffix:str='test', skip_accs: bool=False, distributions: bool=False):
        """
        TODO: make rad freq plots share y axes??
        """
        def xticker(x):
            if isinstance(x, tuple([list, tuple])):
                return [xticker(v) for v in x]
            elif isinstance(x, float):
                return np.log10(x)
            elif isinstance(x, int):
                return x
        def xticklabeller(x):
            if isinstance(x, tuple([list, tuple])):
                return [xticklabeller(v) for v in x]
            elif isinstance(x, float):
                ans =  np.log10(x)
                return f'{ans:.3g}'
            elif isinstance(x, int):
                return f'{x}'
        def xlabeller(x):
            if isinstance(x, float):
                return f'log weight decay'
            else:
                return f'depth'
        for s in sm_args.keys():
            archfams = sm_args[s]
            for a, v in archfams.items():
                print(f'looping over arch. {a} {s} sweep')
                if not skip_accs:
                    acc_dict_df: pd.DataFrame = None
                for i, (pf, w) in enumerate(v.items()):
                    if not skip_accs:
                        if s == 'decay':
                            acc_dict = decay_sweep_acc_dict(
                                sweep_dir = w['acc_dir'],
                                sm_dict=sm, legacy=False)
                            acc_dict = {k: acc_dict[k] 
                                for k in sorted(list(acc_dict.keys()), 
                                    key=lambda x: floating_string(x))}
                        else:
                            acc_dict = depth_sweep_acc_dict(
                                sweep_dir = w['acc_dir'],
                                sm_dict=sm,
                            )
                            acc_dict = {k: acc_dict[k] 
                                for k in sorted(list(acc_dict.keys()), 
                                    key=lambda x: int(x))}
                        acc_dict_means = {k: np.mean(np.array(list(v.values()))) for k, 
                            v in acc_dict.items()}
                        acc_dict_summary = {
                            k: (acc_dict_means[k], np.sqrt(np.var(np.array(list(v.values())) - acc_dict_means[k])))
                            for k, v in acc_dict.items()
                            }
                        print(f'validation accuracies:\n')
                        print(acc_dict_summary)
                        prenewdf = {str(floating_string(k)): [' $+-$ '.join([f'{x:.3f}' for x in v])] 
                                    for k, v in acc_dict_summary.items()}
                        newdf = pd.DataFrame(prenewdf, index = [pf])
                        if i == 0:
                            acc_dict_df = newdf
                        else:
                            acc_dict_df = pd.concat([acc_dict_df, newdf], axis=0)

                    loc_sm = sm_averager(sm[w['sm_key']])
                    # fig, ax = frequency_image(loc_sm, glob_max=False, logparam=True)
                    # fig.tight_layout()
                    # fig.savefig(pp_dir/ f'freq_image_{s}_{pf}_{a}_{suffix}.png',dpi=200)
                    # plt.close(fig)
                    smoothing_radius = 3
                    if re.search('vggi', a):
                        smoothing_radius = 5 
                    # loc_sm = {pf: rp_averager(sm[w['sm_key']], 
                    #     distributions=True, smooth=smoothing_radius, 
                    #     # bin=True,
                    #     ) for pf, w in v.items()}
                    # fig, ax = radial_freq_dist(loc_sm, input_data=True, share=True)
                    # fig.tight_layout()
                    # fig.savefig(pp_dir/ f'radial_{s}_{a}_{suffix}.png',dpi=200)
                    # plt.close(fig)
                    loc_sm = {pf: rp_averager(sm[w['sm_key']], 
                        distributions=distributions, logparam=True, smooth=smoothing_radius, 
                        # bin=True,
                        ) for pf, w in v.items()}
                    # hack to deal w/ untrained nets:
                    if (re.search('decay', s) and (len(list(loc_sm.items()))==1)):
                        include_legend=False
                    else:
                        include_legend=True
                    fig, ax = radial_freq_dist(loc_sm, input_data=True, 
                        ylabel_logscale=True, xscale_log=True, share=True,
                        include_legend=include_legend)
                    fig.tight_layout()
                    fig.savefig(pp_dir/ f'radial_loglog_{s}_{a}_{suffix}.png',dpi=200)
                    plt.close(fig)
                    # recall v['image_stats'] is an output of radial pushforward
                    # so it's imr, ans, maybe ansstd
                    # TODO: doesn't work with binning. sinning?
                    # for divname, div in div_args.items():
                    #     klds = div_averager(sm[v['sm_key']], v['image_stats'][1], 
                    #         div=div)
                    #     klds = {k: v for (k, v) in sorted(list(klds.items()), key= lambda x: float(x[0]))}
                    #     fig, ax = plt.subplots(1,1, figsize=(8,4))
                    #     ax.bar(x=np.arange(len(klds.keys())), height=[klds[k]['meannorm'][0] for k in klds.keys()], yerr=[klds[k]['meannorm'][1] for k in klds.keys()], width=1/4, capsize=4)
                    #     ax.set_xticks(np.arange(len(klds.keys())))
                    #     ax.set_xticklabels(xticklabeller(list(klds.keys())))
                    #     ax.set_ylabel(f'{divname}')
                    #     ax.set_xlabel(xlabeller(list(klds.keys())[0]))
                    #     dn = divname.replace(' ', '_')
                    #     fig.tight_layout()
                    #     fig.savefig(pp_dir/ f'{dn}_{s}_{a}_{suffix}.png',dpi=200)
                    #     plt.close(fig)
                if not skip_accs:
                    with open(pp_dir/ f'{s}_{a}_acc.md', 'w') as adc:
                        adc.write(acc_dict_df.to_markdown())

    if args.unnormalized:
        smplots(sm_args=test_sm_args, div_args=div_args, suffix='test', distributions=False)
        smplots(sm_args=train_sm_args, div_args=div_args, suffix='train', distributions=False)
        smplots(sm_args=untrained_sm_args, div_args=div_args, 
            suffix='untrained', skip_accs=True, distributions=False)
    else:
        smplots(sm_args=untrained_sm_args, div_args=div_args, 
            suffix='untrained', skip_accs=True, distributions=True)
        smplots(sm_args=test_sm_args, div_args=div_args, suffix='test', distributions=True)
        smplots(sm_args=train_sm_args, div_args=div_args, suffix='train', distributions=True)