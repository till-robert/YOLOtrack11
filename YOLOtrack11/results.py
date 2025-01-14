from ultralytics.engine.results import Results,BaseTensor, Keypoints, Boxes
import torch

class ZAxisResults(Results):
     def __init__(
        self, orig_img, path, names, boxes=None, masks=None, probs=None, keypoints=None, obb=None, zaxis=None, speed=None
    ) -> None:
          super().__init__( orig_img, path, names, boxes, masks, probs, keypoints, obb, speed)
        #   self.boxes = Boxes(boxes, self.orig_shape)
        #   self.keypoints = Keypoints(keypoints, self.orig_shape)
          self.zaxis = ZAxis(torch.cat([boxes,zaxis,keypoints.reshape(-1,2)],1), self.orig_shape)
          self._keys = list(self._keys)
          self._keys.append("zaxis")

class ZAxis(BaseTensor):
    pass