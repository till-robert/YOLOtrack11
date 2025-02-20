# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from .results import ZAxisResults as Results
from ultralytics.models.yolo.pose.predict import PosePredictor
from ultralytics.utils import DEFAULT_CFG, ops
from ultralytics.utils.checks import check_imgsz
from ultralytics.data.augment import classify_transforms
from ultralytics.utils import LOGGER
from ultralytics.engine.predictor import STREAM_WARNING
from .dataset import load_inference_source
from .utils import scale_boxes, scale_coords
import numpy as np

class ZAxisPredictor(PosePredictor):
    """
    A class extending the DetectionPredictor class for prediction based on an ZAxis model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.obb import OBBPredictor

        args = dict(model="yolov8n-obb.pt", source=ASSETS)
        predictor = OBBPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes OBBPredictor with optional model and data configuration overrides."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "zaxis"
        overrides["augment"] = False
        self.physical_scale = overrides.get("physical_scale", cfg.get("physical_scale", (1,1,1)))

    def preprocess(self, im):
        """
        Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            if len(im.shape)==4 : im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            if len(im.shape)==3 : im = im[:,None,...] #add channel axis if greyscale
            
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)
        is_16bit = im.dtype == torch.uint16
        im = im.to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        if not_tensor:
            if is_16bit : im /= 2**16-1
            else: im /= 255  # 0 - 255 to 0.0 - 1.0
        return im

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            # nkpt = self.model.kpt_shape[0]
            npar = self.model.model.num_extra_parameters
            pred_kpts = pred[:, 6+npar:].view(len(pred), *self.model.kpt_shape) if len(pred) else pred[:, 6+npar:]
            pred_kpts = scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], zaxis=pred[:, 6:6+npar], keypoints=pred_kpts, physical_scale=self.physical_scale))
        return results
    def setup_source(self, source):
        """Sets up source and inference mode."""
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # check image size
        self.transforms = (
            getattr(
                self.model.model,
                "transforms",
                classify_transforms(self.imgsz[0], crop_fraction=self.args.crop_fraction),
            )
            if self.args.task == "classify"
            else None
        )
        self.dataset = load_inference_source(
            source=source,
            batch=self.args.batch,
            vid_stride=self.args.vid_stride,
            buffer=self.args.stream_buffer,
        )
        self.source_type = self.dataset.source_type
        if not getattr(self, "stream", True) and (
            self.source_type.stream
            or self.source_type.screenshot
            or len(self.dataset) > 1000  # many images
            or any(getattr(self.dataset, "video_flag", [False]))
        ):  # videos
            LOGGER.warning(STREAM_WARNING)
        self.vid_writer = {}