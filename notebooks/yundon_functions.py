import PIL
import numpy as np
import os
from scipy.io import loadmat
from matplotlib.patches import Rectangle, Circle
class YundonImage:
    def __init__(self,image_set):
        files_in_img_set_folder = [entry for entry in os.listdir(image_set) if not entry.startswith(".")]
        tifs = [image_set+"/"+entry for entry in files_in_img_set_folder if entry.endswith(".tif")]
        tifs.sort()
        self.img_objects = [PIL.Image.open(tif) for tif in tifs]
        self.img_lengths = [img_obj.n_frames for img_obj in self.img_objects]
        matfile = [image_set+"/"+entry for entry in files_in_img_set_folder if entry.endswith(".mat")][0]
        self.x,self.y,self.z0 = [v.astype(np.float32) for v in loadmat(matfile)['BugXYZPositionAmpFinal'].T[:3]]
        self.x = ((self.x/0.325))
        self.y = ((self.y/0.325))


    def __getitem__(self, idx):
        if idx < 0 or idx >= sum(self.img_lengths):
            raise IndexError("Index out of range for YundonImage")
            
        # tif, tif_idx = 0,idx
        # while tif_idx >= self.img_lengths[tif]:
        #     tif_idx -= self.img_lengths[tif]
        #     tif += 1
        tif, tif_idx = divmod(idx, 387)
        self.img_objects[tif].seek(tif_idx)
        return self.img_objects[tif]
    def __del__(self):
        for img_obj in self.img_objects:
            img_obj.close()
    def plot_gt(self, ax, z_level):
        z = 0.2676 * z_level - self.z0
        bw = (np.abs(z)/0.161*0.21+55)
        bh = bw.copy()

        z_mask = (z > -110) & (z < 110)
        # z_mask = np.ones_like(z, dtype=bool)
        ax.imshow(self[z_level], vmin=1.95e4, vmax=2.05e4, cmap="gray")
        for x_val,y_val,z_val,bw_val, bh_val in zip(self.x[z_mask],self.y[z_mask],z[z_mask], bw[z_mask], bh[z_mask]):
            rect = Rectangle((x_val-0.5*bw_val,y_val-0.5*bh_val),bw_val,bh_val, linewidth=1, edgecolor="yellow", facecolor='none')
            tx,ty = rect.get_xy()
            ax.add_patch(rect)
            tx+=6
            ty-=12
            circle = Circle((x_val,y_val),1, facecolor="red",edgecolor="red")
            ax.add_patch(circle)
            ax.text(tx,ty,f"z={z_val:.3f}",fontsize="small",bbox=dict(facecolor='white', alpha=0.5,))
        return ax


def gt_yundon(idx):
    x,y,z=np.loadtxt("/home/jupyter-till/Pictures/TrackingResultBugs_4th_TrackingGroundTruthScript250213.csv", delimiter=",").T
    x/=0.325
    y/=0.325

    z -= 0.2676 * idx


    bx = (np.abs(z)/0.161*0.21+55)
    by = (np.abs(z)/0.161*0.21+55)
    bboxes = np.array([x,y,bx,by]).T    
    # z/=210 # convert to network scale
    mask = (z > -102) & (z < 108)
    # if(len(data)==0):
    #      return plot_result(ax,path, np.empty(0),np.empty((0,4)),np.empty(0),np.empty((0,0)))
    cls = np.zeros_like(x)
    
    kpts = np.array([x,y]).T    
    return x[mask]/1280,y[mask]/1080,-z[mask]