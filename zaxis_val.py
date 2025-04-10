import numpy as np
import matplotlib.pyplot as plt

import YOLOtrack11
model = YOLOtrack11.YOLOtrack11("../ultralytics/runs/pose/train79/weights/best.pt")

r,p,xy_rms, z_rms = [],[],[],[]
for i in range(16):
    fname = f'datasets/Dataset_yundon/levels/dataset{i}.yaml'
    res = model.val(conf=0.8,data=fname, z_corr=False)
    r.append(res.box.r)
    p.append(res.box.p)
    xy_rms.append(min(res.xy_rms)*0.65)
    z_rms.append(min(res.z_rms)*400)

    del res

noise_range = [40,50]


test_noiselevels = np.linspace(*noise_range, 11)
# iou_levels = np.linspace(0.5,0.95,10).round(2)
plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.xlabel("noise power")
plt.ylabel("r.m.s xy")
plt.legend(fontsize="small")
plt.plot(test_noiselevels, r, label="Recall")
plt.plot(test_noiselevels, p, label="Precision")

plt.axvline(45,0,1,ls="dashed", lw=0.9, label="training lower noise bound")
plt.axvline(50,0,1,ls="dashed", lw=0.9, label="training upper noise bound")
plt.legend()

plt.subplot(2,2,3)
plt.axvline(45,0,1,ls="dashed", lw=0.9, label="training lower noise bound")
plt.axvline(50,0,1,ls="dashed", lw=0.9, label="training upper noise bound")
plt.plot(test_noiselevels, xy_rms)
plt.xlabel("noise power")
plt.ylabel("xy rms. / um")
plt.subplot(2,2,4)
plt.axvline(45,0,1,ls="dashed", lw=0.9, label="training lower noise bound")
plt.axvline(50,0,1,ls="dashed", lw=0.9, label="training upper noise bound")
plt.plot(test_noiselevels, xy_rms)
plt.xlabel("noise power")
plt.ylabel("z rms. / um")
plt.savefig("noise_levels.pdf")