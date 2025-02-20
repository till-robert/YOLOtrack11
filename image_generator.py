# YOLOv8
import numpy as np
pi = np.pi
import matplotlib.pyplot as plt
import configparser
from skimage.draw import rectangle
from scipy import ndimage
from scipy.special import factorial as fac
from scipy.interpolate import interp1d
from skimage.transform import resize
import h5py



class Object:
    def __init__(self, x, y, label, parameters,theta=None): # , theta
        self.x = x
        self.y = y
        self.theta = theta 
        self.label = label
        self.parameters = parameters
class Ripple(Object):
    def __init__(self, x, y, label, parameters,theta=None): # , theta
        super().__init__(x, y, label, parameters,theta)
        self.z = self.parameters["z"]


downsampled_refstack = (np.load("ripples_downsampled.npy")-2e4)
refstack_shape = 128,128
resize_factor = 4
#centerpoint at:
x0,y0=65,64


def generateImage_parallel(objects, image_size, noise_range, i_range=[1,1],rng=np.random.default_rng(), dtype=None):



    if(type(image_size) is int):
        image = np.zeros([image_size, image_size])
    else:
        image = np.zeros(image_size)

    bboxes = []
    labels = []
    pars = []
    if len(objects)==0:
        if isinstance(noise_range, list):
            snr = rng.uniform(noise_range[0], noise_range[1])             
        else:
            snr = noise_range
        noise_power = snr
        
        # Calculate the standard deviation of the noise
        noise_std = np.sqrt(noise_power)
        
        # Generate Gaussian noise
        noise = np.random.normal(0, noise_std, image.shape)
        image = image + noise
        image = image+(2e4/(2**16-1))
        image = image.clip(0,1)

        # print(intensity)
        return (bboxes, labels, pars, image) 
    
    
    x = np.array([obj.x for obj in objects])
    y = np.array([obj.y for obj in objects])
    n = len(objects)

    i_list, s_list,z_list = np.array(objects[0].parameters)
    intensity = rng.uniform(i_range[0], i_range[1],n) if i_list[0] == 0 else i_list[0]
    # s = int(rng.uniform(s_list[0], s_list[1])) if len(s_list) > 1 else s_list[0] # sigma = rng.uniform(1.5, 3)
    z = np.round(rng.uniform(z_list[0], z_list[1],n)).astype(int) if len(z_list) > 1 else z_list[0]
    ripple = downsampled_refstack[z]#/np.sum(downsampled_refstack[z])
    y1,y2,x1,x2 = np.round(y-256/(resize_factor)).astype(int),np.round(y+256/(resize_factor)).astype(int),np.round(x-256/(resize_factor)).astype(int),np.round(x+256/(resize_factor)).astype(int)
    i1,i2,j1,j2=np.ones(n,dtype=int)*0,np.ones(n,dtype=int)*512//(resize_factor),np.ones(n,dtype=int)*0,np.ones(n,dtype=int)*512//(resize_factor)

    mask = (y1<0)
    i1[mask] = -y1[mask]
    y1[mask]=0
    
    mask = (y2>image_size[0])
    i2[mask] = image_size[0]-y2[mask]
    y2[mask]=image_size[0]

    mask = (x1<0)
    j1[mask] = -x1[mask]
    x1[mask]=0

    mask = (x2>image_size[1])
    j2[mask] = image_size[1]-x2[mask]
    x2[mask]=image_size[1]
    for i,(y1v,y2v,x1v,x2v,i1v,i2v,j1v,j2v) in enumerate(zip(y1,y2,x1,x2,i1,i2,j1,j2)):
        image[y1v:y2v,x1v:x2v] += intensity*ripple[i,i1v:i2v,j1v:j2v] #add patches to image

    bx = by = (np.abs(z-761)*0.21+55)/(resize_factor)

    bboxes = np.array([x-bx,y-by,x+bx,y+by]).T
    labels = (f"Ripple",)*n
    pars = z

    # signal_power = np.mean(image**2)
    
    # Calculate noise power from the desired SNR
    if isinstance(noise_range, list):
        noise_std = rng.uniform(noise_range[0], noise_range[1])             
    else:
        noise_std = noise_range

    
    # Generate Gaussian noise
    noise = np.random.normal(0, noise_std, image.shape)
    image = image + noise
    image = image + 2e4
    image = image.clip(0,2**16-1)

    # print(intensity)
    return (bboxes, labels, pars, image) 



def getRandom_parallel(n_list, image_size, distance, offset, label_list, parameters_list,rng, n_gaussian=False):
    assert len(n_list)==2
    if(n_gaussian):
        n = np.round(rng.normal(*n_list))
        n = n.clip(0).astype(np.int32)
    else:
        n = rng.integers(*n_list) #number of points
    points = np.empty((n,2))
    objects = []
    for i in range(n):
        for _ in range(10000):
            new_point = rng.random(2)
            if type(image_size) is int:
                new_point*=(image_size - 2*offset) + offset
            else:
                new_point[1]*=(image_size[0] - 2*offset) + offset
                new_point[0]*=(image_size[1] - 2*offset) + offset
            if(i == 0 or np.all(((points[:i]-new_point)**2).sum(axis=1)>=distance**2)):
                points[i] = new_point
                random_obj_idx = rng.integers(len(label_list))
                objects.append(Object(*new_point,label_list[random_obj_idx],*parameters_list[random_obj_idx]))
                break
    return objects
    
    
    
            
        