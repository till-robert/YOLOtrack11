
import torch
import sys
import os
import PIL.Image
import numpy as np
from ultralytics.utils import ops
import matplotlib.pyplot as plt

def postprocess(preds, img, orig_imgs):
    """Post-processes predictions and returns a list of Results objects."""
    preds = ops.non_max_suppression(
        preds,
        conf_thres=0.7,
        iou_thres=0.7,
        nc=1,
    )

    if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
        orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

    results = []
    for pred, orig_img, img_path in zip(preds, orig_imgs, (image,)):
        pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        # nkpt = self.model.kpt_shape[0]
        npar = 2
        pred_kpts = pred[:, 6+npar:].view(len(pred), 1,2) if len(pred) else pred[:, 6+npar:]
        pred_kpts = scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
        results.append(Results(orig_img, path=img_path, names={0:"s"}, extra_param_names=["z","i"], boxes=pred[:, :6], zaxis=pred[:, 6:6+npar], keypoints=pred_kpts))
    return results

def predict(x, model):
    y = []  # outputs
    for m in model.model:
        if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
        x = m(x)  # run
        y.append(x)  # save output
    return x

sys.path.append("../..")
sys.path.append("..")
from YOLOtrack11.utils import scale_boxes, scale_coords
from YOLOtrack11.results import ZAxisResults as Results
from YOLOtrack11 import YOLOtrack11

dataset_path = "datasets/Dataset_spots/images/test_snr5.0"
imgsz=640,540
test_images = os.listdir(dataset_path)
random_image = lambda: dataset_path+"/"+test_images[np.random.randint(0,len(test_images))]
image = random_image()
img = np.array(PIL.Image.open(image))
model = torch.load("notebooks/yolo11n_spots_15-20.pt")
torch_img = torch.Tensor(img/2**16).unsqueeze(0).unsqueeze(0).half().to("cuda:0")
m = model["model"].to("cuda:0")
out = predict(torch_img,m)#predict(torch_img,m)
plt.imshow(postprocess(out, torch_img, torch_img)[0].plot())
plt.savefig("tmp.png")