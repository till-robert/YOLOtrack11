import numpy as np
import matplotlib.pyplot as plt

from YOLOtrack11 import YOLOtrack11
i=1

model_paths = ("notebooks/last.pt",)
model = YOLOtrack11(model_paths[0])
model.save("notebooks/test.pt")
