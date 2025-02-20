import ultralytics
import numpy as np
from ultralytics.utils.instance import Bboxes

class Instances(ultralytics.utils.instance.Instances):
    def __init__(self, bboxes, segments=None, keypoints=None,extra_parameters=None, bbox_format="xywh", normalized=True) -> None:

        super().__init__(bboxes,segments, keypoints,bbox_format=bbox_format, normalized=normalized)
        self.extra_parameters = extra_parameters

        if(not isinstance(self.extra_parameters, np.ndarray)):
            print("test")

    def __getitem__(self, index) -> "Instances":

        segments = self.segments[index] if len(self.segments) else self.segments
        keypoints = self.keypoints[index] if self.keypoints is not None else None
        bboxes = self.bboxes[index]
        extra_parameters = self.extra_parameters[index] if self.extra_parameters else None
        bbox_format = self._bboxes.format
        return Instances(
            bboxes=bboxes,
            segments=segments,
            extra_parameters=extra_parameters,
            keypoints=keypoints,
            bbox_format=bbox_format,
            normalized=self.normalized,
        )
    def update(self, bboxes, segments=None, keypoints=None,extra_parameters=None):
        """Updates instance variables."""
        self._bboxes = Bboxes(bboxes, format=self._bboxes.format)
        if segments is not None:
            self.segments = segments
        if keypoints is not None:
            self.keypoints = keypoints
        if extra_parameters is not None:
            self.extra_parameters = extra_parameters
            
    @classmethod
    def concatenate(cls, instances_list, axis=0):

        assert isinstance(instances_list, (list, tuple))
        if not instances_list:
            return cls(np.empty(0))
        assert all(isinstance(instance, Instances) for instance in instances_list)

        if len(instances_list) == 1:
            return instances_list[0]

        use_keypoint = instances_list[0].keypoints is not None
        use_extra_pars = instances_list[0].extra_parameters is not None
        bbox_format = instances_list[0]._bboxes.format
        normalized = instances_list[0].normalized

        cat_boxes = np.concatenate([ins.bboxes for ins in instances_list], axis=axis)
        cat_segments = np.concatenate([b.segments for b in instances_list], axis=axis)
        cat_keypoints = np.concatenate([b.keypoints for b in instances_list], axis=axis) if use_keypoint else None
        cat_extra_parameters = np.concatenate([b.extra_parameters for b in instances_list], axis=axis) if use_extra_pars else None
        assert len(cat_boxes) == len(cat_extra_parameters), "wrong length"
        return cls(cat_boxes, cat_segments, cat_keypoints, cat_extra_parameters,bbox_format, normalized)
    def clip(self,w,h):
        raise RuntimeError("clipping not allowed")
        super().clip(w,h)