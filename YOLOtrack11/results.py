from ultralytics.engine.results import Results,BaseTensor, Keypoints, Boxes
from .utils import scale_to_physical
import torch

class ZAxisResults(Results):
  def __init__(self, orig_img, path, names, boxes=None, masks=None, probs=None, keypoints=None, obb=None, zaxis=None, speed=None, physical_scale=(1,1,1)):
    super().__init__( orig_img, path, names, boxes, masks, probs, keypoints, obb, speed)
    #   self.boxes = Boxes(boxes, self.orig_shape)
    #   self.keypoints = Keypoints(keypoints, self.orig_shape)
    self.zaxis = ZAxis(torch.cat([boxes,zaxis,keypoints.reshape(-1,2)],1), self.orig_shape)
    self.z = zaxis
    self.keypoints = keypoints
    self._keys = list(self._keys)
    self._keys.append("zaxis")
    self.physical_scale = physical_scale

  def to_physical(self):
    return scale_to_physical(self.keypoints, self.z, self.physical_scale, self.orig_img.shape)

class ZAxis(BaseTensor):
  pass