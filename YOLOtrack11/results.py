from ultralytics.engine.results import Results,BaseTensor, Keypoints, Boxes,Masks, Probs, OBB
#from .utils import scale_to_physical
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle, Circle

import numpy as np

class ZAxisResults(Results):
    def __init__(self, orig_img, path, names, extra_param_names, boxes=None, masks=None, probs=None, keypoints=None, obb=None, zaxis=None, speed=None):
        super().__init__( orig_img, path, names, boxes, masks, probs, keypoints, obb, speed)
        #   self.boxes = Boxes(boxes, self.orig_shape)
        #   self.keypoints = Keypoints(keypoints, self.orig_shape)
        self.zaxis = ZAxis(zaxis, self.orig_shape)
        self.keypoints = keypoints
        self.extra_param_names = extra_param_names
        self._keys = list(self._keys)
        self._keys.append("extra_param_names")
        self._keys.append("zaxis")

  # def to_physical(self):
  #   return scale_to_physical(self.keypoints, self.z, self.physical_scale, self.orig_img.shape)
    def summary(self, normalize=False, decimals=5):
        """
        Converts inference results to a summarized dictionary with optional normalization for box coordinates.

        This method creates a list of detection dictionaries, each containing information about a single
        detection or classification result. For classification tasks, it returns the top class and its
        confidence. For detection tasks, it includes class information, bounding box coordinates, and
        optionally mask segments and keypoints.

        Args:
            normalize (bool): Whether to normalize bounding box coordinates by image dimensions. Defaults to False.
            decimals (int): Number of decimal places to round the output values to. Defaults to 5.

        Returns:
            (List[Dict]): A list of dictionaries, each containing summarized information for a single
                detection or classification result. The structure of each dictionary varies based on the
                task type (classification or detection) and available information (boxes, masks, keypoints).

        Examples:
            >>> results = model("image.jpg")
            >>> summary = results[0].summary()
            >>> print(summary)
        """
        # Create list of detection dictionaries
        results = []
        if self.probs is not None:
            class_id = self.probs.top1
            results.append(
                {
                    "name": self.names[class_id],
                    "class": class_id,
                    "confidence": round(self.probs.top1conf.item(), decimals),
                }
            )
            return results

        is_obb = self.obb is not None
        data = self.obb if is_obb else self.boxes
        h, w = self.orig_shape if normalize else (1, 1)
        for i, row in enumerate(data):  # xyxy, track_id if tracking, conf, class_id
            class_id, conf = int(row.cls), round(row.conf.item(), decimals)
            box = (row.xyxyxyxy if is_obb else row.xyxy).squeeze().reshape(-1, 2).tolist()
            xy = {}
            for j, b in enumerate(box):
                xy[f"x{j + 1}"] = round(b[0] / w, decimals)
                xy[f"y{j + 1}"] = round(b[1] / h, decimals)
            result = {"name": self.names[class_id], "class": class_id, "confidence": conf, "box": xy}
            if data.is_track:
                result["track_id"] = int(row.id.item())  # track ID
            if self.masks:
                result["segments"] = {
                    "x": (self.masks.xy[i][:, 0] / w).round(decimals).tolist(),
                    "y": (self.masks.xy[i][:, 1] / h).round(decimals).tolist(),
                }
            if self.keypoints is not None:
                x, y = self.keypoints[i].data[0].cpu()  # torch Tensor
                result["keypoints"] = {
                    "x": (x / w).numpy().round(decimals).tolist(),  # decimals named argument required
                    "y": (y / h).numpy().round(decimals).tolist(),
                }
            if self.zaxis is not None:
                z = self.zaxis[i].data.cpu().numpy().round(decimals).tolist()
                result["z"] = {k:v for k,v in zip(self.extra_param_names, z)}
            results.append(result)

        return results
    def plot(self, show_extra_params=None, vmin=None, vmax=None, scale=120):
        figh,figw = self.orig_img.shape[:2]
        # make a Figure and attach it to a canvas.
        fig = Figure(figsize=(figw//scale,figh//scale), dpi=scale)
        canvas = FigureCanvasAgg(fig)

        # Do some plotting here
        # ax=fig.subplots()
        ax = Axes(fig, [0., 0., 1.,1.])
        fig.add_axes(ax)
        ax.axis("off")

        boxes = self.boxes.cpu()
        kpts = self.keypoints.cpu().data.numpy()
        bboxes = boxes.xywh.numpy()
        cls = boxes.cls.numpy()
        conf = boxes.conf.numpy()
        zaxis = self.zaxis.cpu().data.numpy()
        
        ax.imshow(self.orig_img,cmap="grey",vmin=vmin,vmax=vmax)


        for bbox,kpt,c,z in zip(bboxes,kpts,conf,zaxis):
            if(bbox is not None):
                x,y,w,h = bbox
                rect = Rectangle((x-0.5*w,y-0.5*h),h,w, linewidth=1, edgecolor="yellow", facecolor='none')
                tx,ty = rect.get_xy()
                ax.add_patch(rect)
                tx+=6
                ty-=12
            else:
                tx,ty = kpt
                ty-=40
                tx-=50
            circle = Circle(kpt.squeeze(),1, facecolor="red",edgecolor="red")
            ax.add_patch(circle)
            ax.text(tx,ty, f"{c*100:.0f}%, z = {z[0]:.2f}um", fontsize="small",bbox=dict(facecolor='white', alpha=0.5,))

        # Retrieve a view on the renderer buffer
        ax.set_xlim(0,figw)
        ax.set_ylim(figh,0)
        canvas.draw()
        buf = canvas.buffer_rgba()
        # convert to a NumPy array
        return np.asarray(buf)

class ZAxis(BaseTensor):
    pass