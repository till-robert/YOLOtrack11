import PIL
import numpy as np
yundon_img = (-1,None) #img_idx, imgfile
imgpaths = ["BGC101_H2V2Z1_Ch1_zstack_1_MMStack_Pos0.ome.tif","BGC101_H2V2Z1_Ch1_zstack_1_MMStack_Pos0_1.ome.tif","BGC101_H2V2Z1_Ch1_zstack_1_MMStack_Pos0_2.ome.tif","BGC101_H2V2Z1_Ch1_zstack_1_MMStack_Pos0_3.ome.tif",]
imglengths = [PIL.Image.open("/home/jupyter-till/Pictures/" + imgpath).n_frames for imgpath in imgpaths]
def open_yundon_img(idx):
    global yundon_img
    global imgpaths
    global imglengths
    for i,imglen in enumerate(imglengths): #find image index and relative index
        if idx >= imglen:
            idx -= imglen
        else: break

    if yundon_img[0] != i: #if correct image is not loaded:
        if isinstance(yundon_img[1], PIL.Image.Image): yundon_img[1].close() #close old image
        yundon_img = i,PIL.Image.open("/home/jupyter-till/Pictures/"+imgpaths[i])
    try:
        yundon_img[1].seek(idx)
    except ValueError as e:
        raise ValueError("seek error",idx, e)

    return yundon_img[1]
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