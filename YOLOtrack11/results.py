from ultralytics.engine.results import Results,BaseTensor


class ZAxisResults(Results):
     def __init__(
        self, orig_img, path, names, boxes=None, masks=None, probs=None, keypoints=None, obb=None, zaxis=None, speed=None
    ) -> None:
          super().__init__( orig_img, path, names, boxes, masks, probs, keypoints, obb, speed)
          self.zaxis = ZAxis(zaxis, self.orig_shape) if zaxis is not None else None
          self._keys = list(self._keys)
          self._keys.append("zaxis")

class ZAxis(BaseTensor):
    pass