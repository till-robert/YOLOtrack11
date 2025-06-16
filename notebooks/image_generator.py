"""Synthetic Image Generator for YOLO

Backend for the Image Generator notebook

Original authors(s): *Diptabrata Paul, Martin Fränzl*
Edited by: Till Pfaff

"""
import numpy as np
import numpy.typing as npt
pi = np.pi
import matplotlib.pyplot as plt
import configparser
from skimage.draw import rectangle
from scipy import ndimage
from scipy.special import factorial as fac
from scipy.interpolate import interp1d
from skimage.transform import resize
import h5py
from typing import List, TypedDict, Any, Union, Literal, Tuple, Callable,Dict,Sequence
Number = Union[int, float]
import warnings
import pandas as pd
from itertools import repeat, chain
from numba import njit
# class Object:
#     """Class representing a particle that may be added to the image
#     """
#     def __init__(self, x : float, y : float, label : str, intensity: float = 1, **pars):
#         """Initializes an Object

#         Args:
#             x (float): x position
#             y (float): y position
#             label (str): Object label
#             intensity(float, optional): relative intensity of the object, must be between 0 and 1. Defaults to 1
#             **pars: additional object parameters
#         """
#         self.x = x
#         self.y = y
#         self.label = label
#         self.intensity = intensity
#         self.parameters = pars
# class Ripple(Object):
#     """Shorthand for defining Ripple objects
#     """
#     def __init__(self, x, y, z, parameters):
#         super().__init__(x, y, "Ripple", parameters)
#         self.z = self.parameters["z"] = z


def getRandom(
        parameters: Sequence[Dict[str, Union[Number, str, Sequence[Number]]]],
        image_size: Union[int, Tuple[int, int]],
        distance: float = 0,
        distance_consider_object_size: bool = False,
        offset: float = 0,
        rng: np.random.Generator = np.random.default_rng(),
        max_tries: int = 10000
        ) -> pd.DataFrame:
    """
    Generate a list of random objects with specified properties.

    Args:
        parameters (Sequence[Dict[str, Union[Number, str, Sequence[Number]]]]): A list of dictionaries, each specifying the properties of the objects to generate.
            Each dictionary may look like this:
```python
    {
        "label": "object_type", # Mandatory, string label for the object type
        "n": ["uniform", (min_value, max_value)], # Mandatory, number of objects to generate, follows the same stle as the other parameters

        # Other parameters can be defined as follows:
        "some-parameter1": ["uniform", (min_value, max_value)],  # Uniform distribution
        "some-parameter2": ["gaussian", (mean, std_dev)],  # Gaussian distribution
        "some-parameter3": [my_function, (*args)],  # Custom function with arguments
        "some-parameter4": 0.5,  # Fixed value

        ...
    }
```
        image_size (Union[int, Tuple[int, int]]): Size of the image frame. Can be an integer (square) or a tuple (height, width).
        distance (float, optional): Minimum distance between objects. Defaults to 0.
        distance_consider_object_size (bool, optional): Include the object size (mean of obj. width and height) into the distance calculation. Defaults to False.
        offset (float, optional): Offset from the image border within which objects cannot be placed. Defaults to 0.
        rng (np.random.Generator, optional): Random number generator for reproducibility. Defaults to np.random.default_rng().
        max_tries (int, optional): Maximum number of attempts to place an object while respecting distance and offset. Defaults to 10000.

    Returns:
        pd.DataFrame: A DataFrame containing the list of objects with random positions and parameters.
    """
    if isinstance(image_size, int):
        image_size = (image_size, image_size)



    n_list = [np.round(_chooseParameters(p, rng, keys=["n"])["n"]).clip(0).astype(int) for p in parameters]
        # labels.append(p.pop("label"))



    if distance < 0:
        raise ValueError("distance must be greater than or equal to 0")
    if offset < 0:
        raise ValueError("offset must be greater than or equal to 0")
    if offset > image_size[0] or offset > image_size[1]:
        raise ValueError(f"offset ({offset}) must be less than image_size ({image_size})")
    if max_tries < 1:
        raise ValueError("max_tries must be greater than or equal to 1")


    points = np.zeros((sum(n_list),2))
    obj_sizes  = np.zeros((sum(n_list)))
    
    expanded_index = chain(*[repeat(i, n) for i, n in enumerate(n_list)])
    objects = []
    for i, label_idx in enumerate(expanded_index):
        params = _chooseParameters(parameters[label_idx], rng, ignorekeys=["n"])
        w,h = _get_width_height(params)
        new_obj_size = np.mean((w,h))
        for j in range(max_tries):
            new_point = rng.random(2)
            new_point[1]*=(image_size[0] - 2*offset) + offset
            new_point[0]*=(image_size[1] - 2*offset) + offset

            if distance_consider_object_size:
                distances_sq = ((points[:i]-new_point)**2).sum(axis=1)-obj_sizes[:i]**2-new_obj_size**2
            else:
                distances_sq = ((points[:i]-new_point)**2).sum(axis=1)

            if((i == 0 or np.all(distances_sq>=distance**2))): #found a new point!
                points[i] = new_point
                if distance_consider_object_size: obj_sizes[i] = new_obj_size
                dic = {"label": params["label"], "x":new_point[0], "y":new_point[1], "w": w, "h": h}
                del params["label"] #remove label from parameters, since it is already in the dictionary
                objects.append({**dic, **params})
                break

        else: # if we didn't break, we couldn't find a new point
            raise RuntimeError(
                f"Couldn't place object no. {i} after maxtries={max_tries}. Perhaps you chose a `distance` or `offset` that is too large or you want to place too many objects. Try increasing `max_tries` or decreasing `distance` and `offset`.")
            break
    

      
    objects = pd.DataFrame(objects)
    return objects



def _chooseParameters(parameters: Dict[str, Any], rng: np.random.Generator, keys: List[str] = None, ignorekeys: List["str"] = []) -> Dict[str, float]:
    """Choose parameters for the objects from the given dictionary

    Args:
        parameters (Dict[str, Any]): Dictionary of parameters
        rng (np.random.Generator): Random number generator
        keys (List[str], optional): List of keys to choose parameters for. Defaults to None, which means all parameters will be chosen.
        ignorekeys (List[str], optional): List of keys to ignore. Cannot be used with together with `keys`

    Returns:
        Dict[str, float]: Dictionary of chosen parameters
    """
    params = {}

    if(len(ignorekeys) != 0 and keys is not None):
        raise ValueError("keys and ignorekeys cannot be used together!")


    for key in keys or parameters.keys():
        if key in ignorekeys:
            continue
        if key not in parameters:
            raise ValueError(f"Key {key} not found in parameters")
        if parameters[key] is None:
            raise ValueError(f"Key {key} has no value")
        if isinstance(parameters[key],Sequence) and len(parameters[key]) == 2:
            if isinstance(parameters[key][0],str) and parameters[key][0] == "uniform":
                params[key] = rng.uniform(*parameters[key][1])
            elif isinstance(parameters[key][0],str) and parameters[key][0] == "gaussian":
                params[key] = rng.normal(*parameters[key][1])
            elif isinstance(parameters[key][0],Callable):
                params[key] = parameters[key][0](*parameters[key][1])
            elif (isinstance(parameters[key][0],Number) and isinstance(parameters[key][0],Number)):
                params[key] = rng.uniform(parameters[key][0], parameters[key][1])
            else:
                raise ValueError(f"Invalid parameter formatting for {key}: {parameters[key]}, must be a list of two elements, where the first element is a string and the second element is a list of two numbers")
        elif isinstance(parameters[key],list) and len(parameters[key]) == 1:
            params[key] = parameters[key][0]
        else:
            params[key] = parameters[key]
    return params

def _get_width_height(obj: Dict[str, Any]) -> Tuple[float, float]:
    if obj["label"] == 'Spot':
        bx = by = 2*obj["s"]
    if obj["label"] == 'Ring':                
        bx = by= 2*obj["s"] + obj["r"]
    if obj["label"] == 'Janus':
        bx = by= 2*obj["s"] + obj["r"]
    if obj["label"] == 'Ellipse':
        bx = 2*obj["sx"] # !!!
        by = 2*obj["sy"] # !!!
    if obj["label"] == 'Rod':
        l, w, s = obj["l"], obj["w"], obj["s"]
        bx = l/2 + 2*s
        by = w/2 + 2*s
    if obj["label"] == 'Ripple':
        bx =  by = (np.abs(obj["z"]-761)*0.21+55)/4
    return 2*bx, 2*by
        


def generateImage(
        objects: pd.DataFrame,
        image_size: Union[int, Tuple[int, int]],
        refstack: np.ndarray = None,
        noise: Union[float, List[float]] = None,
        snr: Union[float, List[float]] = None,
        refstack_center: Tuple[float, float] = None,
        rng: np.random.Generator = np.random.default_rng(),
        background: float = 2e4,
        ) -> Tuple[List[np.ndarray], List[str], List[float], np.ndarray]:
    """Generates a synthetic image with the specified objects and parameters
    Args:
        objects (pd.DataFrame): Objects to be added to the image
        image_size (Union[int, Tuple[int, int]]): Size of the image frame. Either `int` for a square size frame or `(y: int,x: int)` for a rectangular image
        refstack (np.ndarray, optional): Reference stack to be used for generating the image.
        noise (Union[float, List[float]], optional): Standard deviation of the Gaussian noise to be added to the image. If a list is provided, it will be randomly selected from the range.
        snr (Union[float, List[float]], optional): Signal-to-noise ratio range. If a list is provided, it will be randomly selected from the range.  If neither `noise` nor `snr` is provided, the image will be generated without noise.
        refstack_center (Tuple[float, float], optional): center point (y,x) for the refstack, if it is not defined, the center of the image will be used. The coordinates must be in the range of the refstack image size.
        rng (np.random.Generator, optional): Random number generator to be used. Defaults to `np.random.default_rng()`.
        background (float, optional): Background intensity of the image. Defaults to 2e4.
    Returns:
        np.ndarray: the generated image
    """


    if not (isinstance(image_size, int) or isinstance(image_size, Sequence) and len(image_size) == 2):
        raise ValueError("image_size must be an int or a tuple of size 2")
    if(isinstance(image_size, int)):
        image_size = (image_size, image_size)
    
    
    
    image = np.zeros(image_size)

    if len(objects)==0:
        if isinstance(noise, Union[float, List]) and snr is None:
            if isinstance(noise, list):
                noise_std = rng.uniform(noise[0], noise[1])
            else:
                noise_std = noise
        elif isinstance(snr, Union[float, List]) and noise is None:
            if isinstance(snr, list):
                noise_std = rng.uniform(snr[0], snr[1])
            else:
                noise_std = snr
        elif noise is not None and snr is not None:
            raise ValueError("Either noise or snr must be provided, not both")
        else:
            noise_std = 0  

        
        
        # Generate Gaussian noise
        noise = np.random.normal(0, noise_std, image.shape)
        image = image + background
        image += noise
        image = image.clip(0,2**16-1)

        return image
    
    # objects = objects.dropna()
    image = place_ripples(image, objects[objects["label"]=="Ripple"], refstack, refstack_center, image_size)
    image = place_others(image, objects[objects["label"]!="Ripple"], image_size)

    # bx = by = (np.abs(z-761)*0.21+55)/(resize_factor)

    # bboxes = np.array([x-bx,y-by,x+bx,y+by]).T
    # labels = (f"Ripple",)*n

    # signal_power = np.mean(image**2)
    
    # Calculate noise power from the desired SNR
    if isinstance(noise, Union[float, List]) and snr is None:
        if isinstance(noise, list):
            noise_std = rng.uniform(noise[0], noise[1])
        else:
            noise_std = noise
    elif isinstance(snr, Union[float, List]) and noise is None:
        if isinstance(snr, list):
            snr = rng.uniform(snr[0], snr[1])
        
        signal = np.max(np.abs(objects["i"])) #max difference from background for each particle
        noise_std = np.mean(signal) / snr
    elif noise is not None and snr is not None:
        raise ValueError("Either noise or snr must be provided, not both")
    else:
        noise_std = 0
    
    # Generate Gaussian noise
    noise = np.random.normal(0, noise_std, image.shape)
    # print("noise:",noise_std, "snr:", snr)
    image += noise
    image += background
    image = image.clip(0,2**16-1)

    # print(intensity)
    info_dict = {
        "noise": noise_std,
        }
    return image, info_dict

        
def place_ripples(image: np.ndarray, ripples: pd.DataFrame, refstack: np.ndarray, refstack_center: Tuple[float, float], image_size: Tuple[int, int]) -> None:
    '''Places ripples in the image based on the provided DataFrame of ripples.'''
    if not isinstance(ripples, pd.DataFrame):
        raise ValueError("ripples must be a pandas DataFrame")
    if len(ripples) == 0:
        return image
    if not all(col in ripples.columns for col in ["x", "y", "z", "i"]):
        raise ValueError("ripples DataFrame must contain columns 'x', 'y', 'z', and 'i'")
    if not isinstance(refstack, np.ndarray) :
        raise ValueError("refstack must be a numpy array")
    if isinstance(refstack, np.ndarray) and len(refstack.shape) != 3:
        raise ValueError("refstack must be a 3D numpy array")
    if isinstance(refstack, np.ndarray) and refstack.shape[1] != refstack.shape[2]:
        raise ValueError("refstack must be a square 3D numpy array")
    if refstack_center is None:
        refstack_center = (refstack.shape[1]//2, refstack.shape[2]//2)
    if not isinstance(refstack_center, tuple) or len(refstack_center) != 2:
        raise ValueError("refstack_center must be a tuple of two floats")
    if refstack_center[0] < 0 or refstack_center[1] < 0 or refstack_center[0] >= refstack.shape[1] or refstack_center[1] >= refstack.shape[2]:
        raise ValueError("refstack_center must be within the bounds of the reference stack")
    
    x,y,z,intensity = ripples[["x","y","z","i"]].to_numpy().T
    n = len(x)

    ripple = _image_blend(refstack[z.astype(int)], refstack[z.astype(int)+1], z-z.astype(int)) #blend adjacent slices with alpha = decimal part of z

    # calculate distance from center to edge of the ripple
    left = refstack_center[1]
    right = refstack.shape[2] - refstack_center[1]
    top = refstack_center[0]
    bottom = refstack.shape[1] - refstack_center[0]
    

    
    # Calculate the coordinates where the ripples will be placed
    y1,y2,x1,x2 = np.floor(y-top) ,np.floor(y+bottom)+1 ,np.floor(x-left) ,np.floor(x+right)+1 #add 1 for subpixel blending

    # Indices for slicing the ripples
    i1,i2,j1,j2=np.ones(n,dtype=int)*0,np.ones(n,dtype=int)*refstack.shape[1]+1,np.ones(n,dtype=int)*0,np.ones(n,dtype=int)*refstack.shape[2]+1 #add 1 for subpixel blending

    # calculate subpixel positions
    sub_x = (x-refstack_center[1]) % 1 # subpixel is always between 0 and 1, 
    sub_y = (y-refstack_center[0]) % 1


    # Boundary checks
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

    # Convert to int
    y1,y2,x1,x2 = y1.astype(int),y2.astype(int),x1.astype(int),x2.astype(int)
    i1,i2,j1,j2 = i1.astype(int),i2.astype(int),j1.astype(int),j2.astype(int)


    for i,(y1v,y2v,x1v,x2v,i1v,i2v,j1v,j2v) in enumerate(zip(y1,y2,x1,x2,i1,i2,j1,j2)):
        # print("image shape:",image[y1v:y2v,x1v:x2v].shape, "ripple shape:", _subpixel_blending(x[i]-x[i].astype(int),y[i]-y[i].astype(int),ripple[i])[i1v:i2v,j1v:j2v].shape)
        
        image[y1v:y2v,x1v:x2v] += intensity[i]*_subpixel_blending(sub_x[i], sub_y[i],ripple[i])[i1v:i2v,j1v:j2v] #add patches to image

    return image

def place_others(image: np.ndarray, objects: pd.DataFrame, image_size: Tuple[int, int]) -> None:
    """Places other objects in the image based on the provided DataFrame of objects."""

    if not isinstance(objects, pd.DataFrame):
        raise ValueError("objects must be a pandas DataFrame")
    if len(objects) == 0:
        return image
    if not all(col in objects.columns for col in ["x", "y", "label"]):
        raise ValueError("objects DataFrame must contain columns 'x', 'y', and 'label'")
    X, Y = np.meshgrid(np.arange(0, image_size[1]), np.arange(0, image_size[0]))
    for idx,obj in objects.iterrows():
        x = obj["x"]
        y = obj["y"]   
        #a = np.random.uniform(i_range[0], i_range[1])
        if obj["label"] == 'Spot':
            i, s = obj["i"], obj["s"]
            image += i*np.exp(-((X-x)**2+(Y-y)**2)/(2*s**2))
        if obj["label"] == 'Ring':                
            i, r , s = obj["i"], obj["r"], obj["s"]
            image += i*np.exp(-(np.sqrt((X-x)**2+(Y-y)**2)-r)**2/(2*s**2))
        if obj["label"] == 'Janus':
            i, r, s = obj["i"], obj["r"], obj["s"]
            if "theta" not in obj or obj["theta"] is None:
                phi = np.random.random()*2*pi
            else:
                phi = obj["theta"]
            Xr = x + np.cos(phi)*(X-x) - np.sin(phi)*(Y-y)
            Yr = y + np.sin(phi)*(X-x) + np.cos(phi)*(Y-y)
            angle = np.nan_to_num(np.arccos((Xr-x)/np.sqrt(((Xr-x)**2+(Yr-y)**2))))/2
            image += np.cos(angle)**2*i*np.exp(-(np.sqrt((X-x)**2+(Y-y)**2)-r)**2/(2*s**2))
        if obj["label"] == 'Ellipse':
            i, sx, sy = obj["i"], obj["sx"], obj["sy"]
            if "theta" not in obj or obj["theta"] is None:
                theta = np.random.random()*2*pi
            else:
                theta = obj["theta"]
            a = np.cos(theta)**2/(2*sx**2) + np.sin(theta)**2/(2*sy**2)
            b = -np.sin(2*theta)/(4*sx**2) + np.sin(2*theta)/(4*sy**2)
            c = np.sin(theta)**2/(2*sx**2) + np.cos(theta)**2/(2*sy**2)
            image += i*np.exp(-(a*(X-x)**2 + 2*b*(X-x)*(Y-y) + c*(Y-y)**2))
        if obj["label"] == 'Rod':
            i, l, w, s = obj["i"], obj["l"], obj["w"], obj["s"]
            if obj["theta"] is None:
                theta = np.random.random()*2*pi
            else:
                theta = obj["theta"]
            image_h, image_w = image.shape
            im = np.zeros([image_w, image_h])
            im[int(image_w/2-w/2):int(-image_w/2+w/2), int(image_h/2-l/2):int(-image_h/2+l/2)] = 1
            im = ndimage.rotate(im, np.degrees(theta), reshape=False, mode='constant')
            im = ndimage.shift(im, (y-int(image_h/2)+0.5, x-int(image_w/2)+0.5))
            im = ndimage.gaussian_filter(im, s)
            im /= im.max()
            image += i*im
    return image

def _image_blend(img1: np.ndarray, img2: np.ndarray, alpha: float) -> np.ndarray:
    """Linear blend of two images together using the specified alpha value

    Args:
        img1 (np.ndarray): first image
        img2 (np.ndarray): second image
        alpha (float): blending factor (0-1)

    Returns:
        np.ndarray: blended image
    """
    if img1.shape != img2.shape:
        raise ValueError("img1 and img2 must have the same shape")
    if np.any(alpha < 0) or np.any(alpha > 1):
        raise ValueError("alpha must be between 0 and 1")
    
    return (img1.T*(1-alpha) + img2.T*alpha).T


def _subpixel_blending(x: float, y:float, img:np.ndarray) -> np.ndarray:
    """Subpixel blending of the image using bilinear interpolation

    Args:
        x (float): x subpixel position (0-1)
        y (float): y subpixel position (0-1)
        img (np.ndarray): image to be blended

    Returns:
        np.ndarray: blended image
    """
    if x < 0 or y < 0 or x > 1 or y > 1:
        raise ValueError("x and y must be between 0 and 1 but is x={}, y={}".format(x,y))
    if img.ndim != 2:
        raise ValueError("img must be a 2D array")
    
    img_shape = img.shape
    blended_image = np.zeros((img_shape[0]+1, img_shape[1]+1))
    blended_image[:-1, :-1] = img

    blended_image = (1-x)*blended_image+x*np.roll(blended_image, shift=1, axis=1)
    blended_image = (1-y)*blended_image+y*np.roll(blended_image, shift=1, axis=0)

    return blended_image