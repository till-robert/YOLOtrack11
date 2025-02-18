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


downsampled_refstack = (np.load("ripples_downsampled.npy")-2e4)/(2**16-1)
refstack_shape = 128,128
resize_factor = 4
#centerpoint at:
x0,y0=65,64

def generateImage(objects, image_size, snr_range, i_range=[1,1],rng=np.random.default_rng(), dtype=None):
    image = np.zeros([image_size, image_size])
    bboxes = []
    labels = []
    pars = []
    X, Y = np.meshgrid(np.arange(0, image_size), np.arange(0, image_size))



    for obj in objects:
        x = obj.x
        y = obj.y
        #a = rng.uniform(i_range[0], i_range[1])
        if obj.label == 'Spot':
            i_list, s_list = np.array(obj.parameters)
            i = rng.uniform(i_range[0], i_range[1]) if i_list[0] == 0 else i_list[0]
            s = rng.uniform(s_list[0], s_list[1]) if len(s_list) > 1 else s_list[0] # sigma = rng.uniform(1.5, 3)
            image = image + i*np.exp(-((X-x)**2+(Y-y)**2)/(2*s**2))
            bx = 2*s
            by = 2*s
            bboxes.append(np.array([[x-bx,y-by],[x+bx,y+by]]))
            labels.append(obj.label)


        if obj.label == 'Ring':                
            i_list, r_list, s_list = np.array(obj.parameters)
            i = rng.uniform(i_range[0], i_range[1]) if i_list[0] == 0 else i_list[0]
            r = rng.uniform(r_list[0], r_list[1]) if len(r_list) > 1 else r_list[0] # r = rng.uniform(7, 9)
            s = rng.uniform(s_list[0], s_list[1]) if len(s_list) > 1 else s_list[0]      
            image = image + i*np.exp(-(np.sqrt((X-x)**2+(Y-y)**2)-r)**2/(2*s**2))
            bx = 2*s + r
            by = 2*s + r
            bboxes.append(np.array([[x-bx,y-by],[x+bx,y+by]]))
            labels.append(obj.label)
        if obj.label == 'Janus':
            i_list, r_list, s_list = np.array(obj.parameters)
            i = rng.uniform(i_range[0], i_range[1]) if i_list[0] == 0 else i_list[0]
            r = rng.uniform(r_list[0], r_list[1]) if len(r_list) > 1 else r_list[0] # r = rng.uniform(7, 9)
            s = rng.uniform(s_list[0], s_list[1]) if len(s_list) > 1 else s_list[0]
            if obj.theta is None:
                phi = rng.random()*2*pi
            else:
                phi = obj.theta
            Xr = x + np.cos(phi)*(X-x) - np.sin(phi)*(Y-y)
            Yr = y + np.sin(phi)*(X-x) + np.cos(phi)*(Y-y)
            angle = np.nan_to_num(np.arccos((Xr-x)/np.sqrt(((Xr-x)**2+(Yr-y)**2))))/2
            image = image + np.cos(angle)**2*i*np.exp(-(np.sqrt((X-x)**2+(Y-y)**2)-r)**2/(2*s**2))
            bx = 2*s + r
            by = 2*s + r
            bboxes.append(np.array([[x-bx,y-by],[x+bx,y+by]]))
            labels.append(obj.label)
        if obj.label == 'Ellipse':
            i_list, sx_list, sy_list = np.array(obj.parameters)
            i = rng.uniform(i_range[0], i_range[1]) if i_list[0] == 0 else i_list[0]
            sx = rng.uniform(sx_list[0], sx_list[1]) if len(sx_list) > 1 else sx_list[0] # r = rng.uniform(7, 9)
            sy = rng.uniform(sy_list[0], sy_list[1]) if len(sy_list) > 1 else sy_list[0]
            if obj.theta is None:
                theta = rng.uniform(0, pi) 
            else:
                theta = obj.theta
            a = np.cos(theta)**2/(2*sx**2) + np.sin(theta)**2/(2*sy**2)
            b = -np.sin(2*theta)/(4*sx**2) + np.sin(2*theta)/(4*sy**2)
            c = np.sin(theta)**2/(2*sx**2) + np.cos(theta)**2/(2*sy**2)
            image = image + i*np.exp(-(a*(X-x)**2 + 2*b*(X-x)*(Y-y) + c*(Y-y)**2))
            bx = 2*(np.abs(np.cos(theta))*sx + np.abs(np.sin(theta))*sy)
            by = 2*(np.abs(np.sin(theta))*sx + np.abs(np.cos(theta))*sy)
            bboxes.append(np.array([[x-bx,y-by],[x+bx,y+by]]))
            labels.append(obj.label)
        if obj.label == 'Rod':
            i_list, l_list, w_list, s_list = np.array(obj.parameters)
            i = rng.uniform(i_range[0], i_range[1]) if i_list[0] == 0 else i_list[0]
            l = rng.uniform(l_list[0], l_list[1]) if len(l_list) > 1 else l_list[0] # r = rng.uniform(7, 9)
            w = rng.uniform(w_list[0], w_list[1]) if len(w_list) > 1 else w_list[0] 
            s = rng.uniform(s_list[0], s_list[1]) if len(s_list) > 1 else s_list[0]
            if obj.theta is None:
                theta = rng.uniform(0, 2*pi) 
            else:
                theta = obj.theta
            im = np.zeros([image_size, image_size])
            im[int(image_size/2-w/2):int(-image_size/2+w/2), int(image_size/2-l/2):int(-image_size/2+l/2)] = 1
            im = ndimage.rotate(im, np.degrees(theta), reshape=False, mode='constant')
            im = ndimage.shift(im, (y-int(image_size/2)+0.5, x-int(image_size/2)+0.5))
            im = ndimage.gaussian_filter(im, s)
            im /= im.max()
            image = image + i*im
            sx = l + 2*s
            sy = w + 2*s
            bx = (np.abs(np.cos(theta))*sx + np.abs(np.sin(theta))*sy)/2
            by = (np.abs(np.sin(theta))*sx + np.abs(np.cos(theta))*sy)/2
            bboxes.append(np.array([[x-bx,y-by],[x+bx,y+by]]))
            labels.append(obj.label)

    # Set the SNR 
    # image -= image.min()
    noise = rng.normal(0,1,(image_size, image_size))
    # noise = noise/np.var(noise)
    if isinstance(snr_range, list):
        snr = rng.uniform(snr_range[0], snr_range[1])             
    else:
        snr = snr_range
    image = image + noise/snr
    image= image+2e4/(2**16-1)
    image = image.clip(0,1)
    # image = image/(image.max())

    return (bboxes, labels, pars, image) 
def generateImage_parallel(objects, image_size, noise_range, i_range=[1,1],rng=np.random.default_rng(), dtype=None):

    if(not all(obj.label == "Ripple" for obj in objects)):
        return generateImage(objects, image_size, noise_range, i_range,rng, dtype)

    image = np.zeros([image_size, image_size])
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
    
    mask = (y2>image_size)
    i2[mask] = image_size-y2[mask]
    y2[mask]=image_size

    mask = (x1<0)
    j1[mask] = -x1[mask]
    x1[mask]=0

    mask = (x2>image_size)
    j2[mask] = image_size-x2[mask]
    x2[mask]=image_size
    for i,(y1v,y2v,x1v,x2v,i1v,i2v,j1v,j2v) in enumerate(zip(y1,y2,x1,x2,i1,i2,j1,j2)):
        image[y1v:y2v,x1v:x2v] += intensity*ripple[i,i1v:i2v,j1v:j2v] #add patches to image

    bx = by = (np.abs(z-761)*0.21+55)/(resize_factor)

    bboxes = np.array([x-bx,y-by,x+bx,y+by]).T
    labels = (f"Ripple",)*n
    pars = z

    # signal_power = np.mean(image**2)
    
    # Calculate noise power from the desired SNR
    if isinstance(noise_range, list):
        noise_power = rng.uniform(noise_range[0], noise_range[1])             
    else:
        noise_power = noise_range
    
    # Calculate the standard deviation of the noise
    noise_std = np.sqrt(noise_power)
    
    # Generate Gaussian noise
    noise = np.random.normal(0, noise_std, image.shape)
    image = image + noise
    image = image+(2e4/(2**16-1))
    image = image.clip(0,1)

    # print(intensity)
    return (bboxes, labels, pars, image) 


def getRandom(n_list, image_size, distance, offset, label_list, parameters_list,rng):
    '''
    :n_list: int list, # of particles of each class
    :image_size: int, shape of output image
    :distance: int, min distance between points
    :offset: boundary padding
    :label_list: str list, class labels
    :parameters_list: list of parameters for each class 
    '''
    if not isinstance(n_list, list): 
        n_list = [n_list]
    if not isinstance(label_list, list):
        label_list = [label_list]
    if len(n_list) != len(label_list):
        raise ValueError('The lists must have equal length')
    
    points = []
    while len(points) < np.sum(n_list):
        # Generate a random point
        new_point = rng.random(2)*(image_size - 2*offset) + offset
        
        # Check if the new point meets the minimum distance requirement
        if all(np.linalg.norm(np.array(new_point) - np.array(p)) >= distance for p in points):
            points.append(new_point.tolist())
    points = np.array(points)

    objects = []
    j=0
    for i in range(len(parameters_list)):
        objects.append([Object(x, y, label, parameters) for (x, y), label, parameters in zip(points[j:j + n_list[i]], 
                                                                                             np.repeat(label_list[i], n_list[i]).tolist(), 
                                                                                             np.repeat(np.array(parameters_list[i]), n_list[i], axis=0).tolist())])
        j+=n_list[i]

    all_objects = []   
    for obj in objects:
        all_objects.extend(obj)

    return np.array(all_objects)
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
            new_point = rng.random(2)*(image_size - 2*offset) + offset
            if(i == 0 or np.all(((points[:i]-new_point)**2).sum(axis=1)>=distance**2)):
                points[i] = new_point
                random_obj_idx = rng.integers(len(label_list))
                objects.append(Object(*new_point,label_list[random_obj_idx],*parameters_list[random_obj_idx]))
                break
    return objects
    
    
    
            
        