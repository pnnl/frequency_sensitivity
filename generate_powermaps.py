from freq_sens import *
from plotting import *
from datasets import *
from torchvision.transforms import ToPILImage
topil = ToPILImage()
import re
device = ch.device('cuda:0')

covdir = pa.Path('./results/freq-sens/covariances')
dldict = {
    'cifar': cifardl, 
    'imagenette': imagenettedl
}
alphas = [0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
for a in alphas:
    dldict[f'lwn_{floating_string(a)}'] = (lwndl, a)

if __name__=='__main__':
    for k, d in dldict.items():
        if isinstance(d, tuple):
            dls = d[0](alpha=d[1], batch_size=8, num_workers=1, half=False, subsample=5000)
        else:
            if re.search('imagenette', k):
                dls = d(batch_size=8, num_workers=1, half=False)
            else:
                dls = d(batch_size=8, num_workers=1, half=False, subsample=5000)
        for l, dl in dls.items():
            print(f'computing {k} {l} power map')
            p = empirical_power(
                    dl,
                    device=device,
                    progress=True
            )
            ch.save(p, covdir/ f'{k}_{l}_power.pt')

    for k, d in dldict.items():
        if re.search('lwn', k):
            continue
        else:
            if re.search('imagenette', k):
                dls = d(batch_size=8, num_workers=1, 
                    high_pass=0.0714, 
                    half=False)
            else:
                dls = d(batch_size=8, num_workers=1, 
                    high_pass=0.5, 
                    half=False, subsample=5000)
            for l, dl in dls.items():
                print(f'computing {k} {l} power map')
                p = empirical_power(
                        dl,
                        device=device,
                        progress=True
                )
                ch.save(p, covdir/ f'{k}_{l}_power_hp.pt')