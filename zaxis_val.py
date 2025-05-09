import numpy as np
import matplotlib.pyplot as plt

import YOLOtrack11
model = YOLOtrack11.YOLOtrack11("../ultralytics/runs/pose/train122/weights/best.pt")
i=1
fname = f'datasets/Dataset_subpixel/levels/dataset{i}.yaml'
res = model.val(conf=0.8,data=fname, z_corr=False)
print("ok")