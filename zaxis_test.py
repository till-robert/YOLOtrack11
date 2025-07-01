import sys
import numpy as np
import os
import matplotlib.pyplot as plt
sys.path.append("..")
from YOLOtrack11 import YOLOtrack11
# model = YOLOtrack11("yolo11n-yundon.pt")
# print(model.model.model)
print("loaded")
dataset_path = "datasets/Dataset_yundon/images/test"
# dataset_path = "../ultralytics/data_gen/Dataset_hard/images/val"
imgsz=1024,824
# imgsz=640,540
test_images = os.listdir(dataset_path)
random_image = lambda: dataset_path+"/"+test_images[np.random.randint(0,len(test_images))]

from itertools import repeat
from matplotlib.patches import Rectangle, Circle
import PIL.Image
def plot_gt(path,ax,imgsz=[512,512],vmin=1.95e4,vmax=2.05e4):
    data=np.atleast_2d(np.loadtxt(path.replace("images", "labels").replace("jpg","txt").replace("tif","txt"))).T
    # print(data)
    if(len(data)==0):
         return plot_result(ax,path, np.empty(0),np.empty((0,4)),np.empty(0),np.empty((0,0)),vmin=vmin,vmax=vmax)
    cls = data[0]
    bboxes = data[1:5].T*(imgsz*2)
    bboxes[:,2:]=bboxes[:,2:]
    z = data[5]
    kpts = data[6:].T*imgsz
    # print(bboxes)
    
    return plot_result(ax,path, cls,bboxes,z,kpts, vmin=vmin,vmax=vmax)
def plot_result(ax,img,cls,bboxes=repeat(None),z=None,kpts=None, conf=None,vmin=1.95e4,vmax=2.05e4):
    is_conf = conf is not None
    if not is_conf:
         conf = np.zeros_like(cls)
    if isinstance(img, PIL.Image.Image):
        pass
    elif(type(img) == str):
        img = PIL.Image.open(img)
    print(vmin,vmax)
    ax.imshow(img,cmap="grey",vmin=vmin,vmax=vmax)
    ax.axis("off")
    for bbox,z_value,kpt,c in zip(bboxes,z, kpts,conf):
        if(bbox is not None):
            x,y,w,h = bbox
            rect = Rectangle((x-0.5*w,y-0.5*h),h,w, linewidth=1, edgecolor="yellow", facecolor='none')
            tx,ty = rect.get_xy()
            ax.add_patch(rect)
            tx+=6
            ty-=12
        else:
            tx,ty = kpt
            ty-=40
            tx-=50
        circle = Circle(kpt,1, facecolor="red",edgecolor="red")
        ax.add_patch(circle)
        ax.text(tx,ty,f"z={z_value:.3f}" + (f", {c*100:.0f}%" if is_conf else ""),fontsize="small",bbox=dict(facecolor='white', alpha=0.5,))

    return bboxes,z

img = "datasets/Zstack_DownSampled_BGCorrected_TrackingData/Image Set 8 - 250616"
finetuned_model = YOLOtrack11("notebooks/yolo11n-yundon_pretrained.pt")
res = finetuned_model.predict(img, conf=0.5, background=2e4, show=True)
# fig=plt.figure(2)
# plt.clf()
# fig.set_figheight(5)
# fig.set_figwidth(10)
# plt.subplot(121)
# plt.title("prediction")
# res[0].plot(plt.gca(),vmin=1.95e4, vmax=2.05e4)
# plt.axis("off")
# plt.subplot(122)
# plt.title("ground truth")
# plot_gt(img, plt.gca(), (1024,824))
# # plt.axis("on")
# plt.show()