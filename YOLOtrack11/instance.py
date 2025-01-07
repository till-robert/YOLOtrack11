import ultralytics
import numpy as np
from ultralytics.utils.instance import Bboxes

class Instances(ultralytics.utils.instance.Instances):
    def __init__(self, bboxes, segments=None, keypoints=None,z_positions=None, bbox_format="xywh", normalized=True) -> None:

        super().__init__(bboxes,segments, keypoints,bbox_format=bbox_format, normalized=normalized)
        self.z_positions = z_positions
    def __getitem__(self, index) -> "Instances":

        segments = self.segments[index] if len(self.segments) else self.segments
        keypoints = self.keypoints[index] if self.keypoints is not None else None
        bboxes = self.bboxes[index]
        z_positions = self.z_positions[index] if self.z_positions is not None else None
        bbox_format = self._bboxes.format
        return Instances(
            bboxes=bboxes,
            segments=segments,
            z_positions=z_positions,
            keypoints=keypoints,
            bbox_format=bbox_format,
            normalized=self.normalized,
        )
    def update(self, bboxes, segments=None, keypoints=None,z_positions=None):
        """Updates instance variables."""
        self._bboxes = Bboxes(bboxes, format=self._bboxes.format)
        if segments is not None:
            self.segments = segments
        if keypoints is not None:
            self.keypoints = keypoints
        if z_positions is not None:
            self.z_positions = z_positions
            
    @classmethod
    def concatenate(cls, instances_list, axis=0):

        assert isinstance(instances_list, (list, tuple))
        if not instances_list:
            return cls(np.empty(0))
        assert all(isinstance(instance, Instances) for instance in instances_list)

        if len(instances_list) == 1:
            return instances_list[0]

        use_keypoint = instances_list[0].keypoints is not None
        use_zaxis = instances_list[0].z_positions is not None
        bbox_format = instances_list[0]._bboxes.format
        normalized = instances_list[0].normalized

        cat_boxes = np.concatenate([ins.bboxes for ins in instances_list], axis=axis)
        cat_segments = np.concatenate([b.segments for b in instances_list], axis=axis)
        cat_keypoints = np.concatenate([b.keypoints for b in instances_list], axis=axis) if use_keypoint else None
        cat_z_positions = np.concatenate([b.z_positions for b in instances_list], axis=axis) if use_zaxis else None
        assert len(cat_boxes) == len(cat_z_positions), "wrong length"
        return cls(cat_boxes, cat_segments, cat_keypoints, cat_z_positions,bbox_format, normalized)