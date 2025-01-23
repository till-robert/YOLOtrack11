# Ultralytics YOLO 🚀, AGPL-3.0 license

import random

import cv2
import numpy as np
import torch
import ultralytics

from ultralytics.data.augment import Mosaic,CopyPaste,Compose, MixUp, Albumentations, RandomFlip,RandomHSV
from ultralytics.utils import LOGGER
import ultralytics.utils
from ultralytics.utils.ops import xyxyxyxy2xywhr

from .instance import Instances

class RandomPerspective(ultralytics.data.augment.RandomPerspective):
    def __call__(self, labels):
        if self.pre_transform and "mosaic_border" not in labels:
            labels = self.pre_transform(labels)
        labels.pop("ratio_pad", None)  # do not need ratio pad

        img = labels["img"]
        cls = labels["cls"]
        instances = labels.pop("instances")
        # Make sure the coord formats are right
        instances.convert_bbox(format="xyxy")
        instances.denormalize(*img.shape[:2][::-1])

        border = labels.pop("mosaic_border", self.border)
        self.size = img.shape[1] + border[1] * 2, img.shape[0] + border[0] * 2  # w, h
        # M is affine matrix
        # Scale for func:`box_candidates`
        img, M, scale = self.affine_transform(img, border)
        # print(bboxes.shape)
        bboxes = self.apply_bboxes(instances.bboxes, M)

        segments = instances.segments
        keypoints = instances.keypoints
        z_positions = instances.z_positions
        # Update bboxes if there are segments.
        if len(segments):
            bboxes, segments = self.apply_segments(segments, M)

        if keypoints is not None:
            keypoints = self.apply_keypoints(keypoints, M)
        new_instances = Instances(bboxes, segments, keypoints, z_positions,bbox_format="xyxy", normalized=False)
        # Clip
        # new_instances.clip(*self.size)

        # Filter instances
        instances.scale(scale_w=scale, scale_h=scale, bbox_only=True)
        # Make the bboxes have the same scale with new_bboxes
        i = self.box_candidates(
            box1=instances.bboxes.T, box2=new_instances.bboxes.T, area_thr=0.01 if len(segments) else 0.10
        )
        labels["instances"] = new_instances[i]
        labels["cls"] = cls[i]
        labels["img"] = img
        labels["resized_shape"] = img.shape[:2]
        return labels

class LetterBox(ultralytics.data.augment.LetterBox):
    def __call__(self, labels=None, image=None):
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        value = 2**16 /2 if img.dtype == np.uint16 else 128.
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(value,)*3
        )  # add border
        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (labels["ratio_pad"], (left, top))  # for evaluation
        
        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img

class Format(ultralytics.data.augment.Format):
    def __init__(
        self,
        bbox_format="xywh",
        normalize=True,
        return_mask=False,
        return_keypoint=False,
        return_obb=False,
        return_zaxis=False,
        mask_ratio=4,
        mask_overlap=True,
        batch_idx=True,
        bgr=0.0,
    ):

        super().__init__(bbox_format, normalize, return_mask, return_keypoint, return_obb, mask_ratio, mask_overlap, batch_idx, bgr)
        self.return_zaxis = return_zaxis

    def __call__(self, labels):

        img = labels.pop("img")
        h, w = img.shape[:2]
        cls = labels.pop("cls")
        #z_position = labels.pop("z")
        instances = labels.pop("instances")
        instances.convert_bbox(format=self.bbox_format)
        instances.denormalize(w, h)
        nl = len(instances)

        if self.return_mask:
            if nl:
                masks, instances, cls = self._format_segments(instances, cls, w, h)
                masks = torch.from_numpy(masks)
            else:
                masks = torch.zeros(
                    1 if self.mask_overlap else nl, img.shape[0] // self.mask_ratio, img.shape[1] // self.mask_ratio
                )
            labels["masks"] = masks
        labels["img"] = self._format_img(img)
        labels["cls"] = torch.from_numpy(cls) if nl else torch.zeros(nl)
        #labels["z_pos"] = torch.from_numpy(z_position) if nl else torch.zeros(nl)
        labels["bboxes"] = torch.from_numpy(instances.bboxes) if nl else torch.zeros((nl, 4))
        if self.return_keypoint or self.return_zaxis:
            labels["keypoints"] = torch.from_numpy(instances.keypoints)
            if self.normalize:
                labels["keypoints"][..., 0] /= w
                labels["keypoints"][..., 1] /= h
        if self.return_obb:
            labels["bboxes"] = (
                xyxyxyxy2xywhr(torch.from_numpy(instances.segments)) if len(instances.segments) else torch.zeros((0, 5))
            )
        if self.return_zaxis:
            labels["extra_parameters"] = torch.from_numpy(instances.extra_parameters)
        # NOTE: need to normalize obb in xywhr format for width-height consistency
        if self.normalize:
            labels["bboxes"][:, [0, 2]] /= w
            labels["bboxes"][:, [1, 3]] /= h
        # Then we can use collate_fn
        if self.batch_idx:
            labels["batch_idx"] = torch.zeros(nl)
        return labels

    def _format_img(self, img):
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img[::-1] if random.uniform(0, 1) > self.bgr else img)
        img = torch.from_numpy(img.copy())
        return img




def v8_transforms(dataset, imgsz, hyp, stretch=False):
    # mosaic = Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic)
    # affine = RandomPerspective(
    #     degrees=hyp.degrees,
    #     translate=hyp.translate,
    #     scale=hyp.scale,
    #     shear=hyp.shear,
    #     perspective=hyp.perspective,
    #     pre_transform=None if stretch else LetterBox(new_shape=(imgsz, imgsz)),
    # )

    # pre_transform = Compose([mosaic, affine])
    # if hyp.copy_paste_mode == "flip":
    #     pre_transform.insert(1, CopyPaste(p=hyp.copy_paste, mode=hyp.copy_paste_mode))
    # else:
    #     pre_transform.append(
    #         CopyPaste(
    #             dataset,
    #             pre_transform=Compose([Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic), affine]),
    #             p=hyp.copy_paste,
    #             mode=hyp.copy_paste_mode,
    #         )
    #     )
    flip_idx = dataset.data.get("flip_idx", [])  # for keypoints augmentation
    if dataset.use_keypoints:
        kpt_shape = dataset.data.get("kpt_shape", None)
        if len(flip_idx) == 0 and hyp.fliplr > 0.0:
            hyp.fliplr = 0.0
            LOGGER.warning("WARNING ⚠️ No 'flip_idx' array defined in data.yaml, setting augmentation 'fliplr=0.0'")
        elif flip_idx and (len(flip_idx) != kpt_shape[0]):
            raise ValueError(f"data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}")
    return Compose(
        [
            # pre_transform,
            # MixUp(dataset,pre_transform=pre_transform, p=hyp.mixup),
            # Albumentations(p=1.0),
            # RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            RandomFlip(direction="vertical", p=hyp.flipud),
            RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx),
        ]
    )  # transforms

