import matplotlib.pyplot as plt
import numpy as np
import torch

from scipy.stats import binned_statistic,gaussian_kde
import seaborn
from importlib import reload
import YOLOtrack11

model = YOLOtrack11.YOLOtrack11("../ultralytics/runs/pose/train122/weights/last.pt")
snr_range = [1,18]
snrlevels = np.linspace(*snr_range, 18)
r,xy_rms, z_rms = [],[],[]
for i, snr in enumerate(snrlevels):
    fname = f'datasets/Dataset_subpixel/levels/dataset{i}.yaml'
    res = model.val(conf=0.7,data=fname, z_corr=False, verbose=True)
    r.append(res.all_box_results)
    xy_rms.append(res.xy_rms)
    z_rms.append(res.z_rms)

    del res
    torch.cuda.empty_cache()