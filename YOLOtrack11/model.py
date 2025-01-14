import torch
import torch.nn as nn
import ultralytics.cfg
from ultralytics.utils import LOGGER,colorstr
from ultralytics.nn.tasks import DetectionModel,initialize_weights, BaseModel, yaml_model_load
from ultralytics.nn.modules import (
    AIFI, C1, C2, C2PSA, C3, C3TR, ELAN1, OBB, PSA, SPP, SPPELAN, SPPF, AConv, ADown, Bottleneck, BottleneckCSP, C2f, C2fAttn, C2fCIB, C2fPSA, C3Ghost, C3k2, C3x, CBFuse, CBLinear, Classify, Concat, Conv, Conv2, ConvTranspose, Detect, DWConv, DWConvTranspose2d, Focus, GhostBottleneck, GhostConv, HGBlock, HGStem, ImagePoolingAttn, Pose, RepC3, RepConv, RepNCSPELAN4, RepVGGDW, ResNetLayer, RTDETRDecoder, SCDown, Segment, WorldDetect, v10Detect,)
from copy import deepcopy
import ultralytics.utils.checks
from pathlib import Path
from ultralytics.utils.ops import make_divisible


from .loss import v8ZAxisLoss
from .utils import scale_img






class ZAxis(Pose):
    """YOLO ZAxis detection head for detection with 3D models."""

    def __init__(self, nc=80, num_extra_parameters=1, kpt_shape=(1,2), ch=()):
        """Initialize ZAxis with number of classes `nc` and layer channels `ch`."""
        super().__init__(nc,kpt_shape, ch)
        self.kpt_branch = self.cv4
        self.ne = num_extra_parameters  # number of extra parameters

        c4 = max(ch[0] // 4, self.ne)
        self.z_branch = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)


        # self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        # self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total

        # c4 = max(ch[0] // 4, self.nk)
        # self.kpt_branch = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        bs = x[0].shape[0]  # batch size
        z = torch.cat([self.z_branch[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)  # z theta logits
        kpt = torch.cat([self.kpt_branch[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)


        # NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
        z = z.sigmoid() # z = [0,1]
        if not self.training:
            self.z = z
        x = Detect.forward(self, x)
        if self.training:
            return x, z, kpt
        pred_kpt = self.kpts_decode(bs, kpt)

        return torch.cat([x, z,pred_kpt], 1) if self.export else (torch.cat([x[0],z, pred_kpt], 1), (x[1],z, kpt))

    # def decode_bboxes(self, bboxes, anchors): not needed for for z-axis detection
    #     """Decode rotated bounding boxes."""
    #     return dist2rbox(bboxes, self.angle, anchors, dim=1)


class ZAxisModel(DetectionModel):
    """YOLOv8 Z-Axis model."""

    def __init__(self, cfg="YOLOtrackv11.yaml", ch=1, nc=None, num_extra_parameters=1, kpt_shape=[], verbose=True):
        """Initialize YOLOv8 Z-Axis model with given config and parameters."""

        BaseModel.__init__(self)
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        if any(kpt_shape) and list(kpt_shape) != list(cfg["kpt_shape"]):
            LOGGER.info(f"Overriding model.yaml kpt_shape={cfg['kpt_shape']} with kpt_shape={kpt_shape}")
            cfg["kpt_shape"] = kpt_shape
        cfg["num_extra_parameters"] = num_extra_parameters

        if self.yaml["backbone"][0][2] == "Silence":
            LOGGER.warning(
                "WARNING ⚠️ YOLOv9 `Silence` module is deprecated in favor of nn.Identity. "
                "Please delete local *.pt file and re-download the latest model checkpoint."
            )
            self.yaml["backbone"][0][2] = "nn.Identity"

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.inplace = self.yaml.get("inplace", True)
        self.end2end = getattr(self.model[-1], "end2end", False)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect
            s = 256  # 2x min stride
            m.inplace = self.inplace

            def _forward(x):
                """Performs a forward pass through the model, handling different Detect subclass types accordingly."""
                if self.end2end:
                    return self.forward(x)["one2many"]
                return self.forward(x)[0] if isinstance(m, (Segment, Pose, OBB, ZAxis)) else self.forward(x)

            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

    def init_criterion(self):
        """Initialize the loss criterion for the model."""
        return v8ZAxisLoss(self)
    
    # def _predict_augment(self, x):
    #     """Perform augmentations on input image x and return augmented inference and train outputs."""
    #     if getattr(self, "end2end", False) or self.__class__.__name__ != "DetectionModel":
    #         LOGGER.warning("WARNING ⚠️ Model does not support 'augment=True', reverting to single-scale prediction.")
    #         return self._predict_once(x)
    #     img_size = x.shape[-2:]  # height, width
    #     s = [1, 0.83, 0.67]  # scales
    #     f = [None, 3, None]  # flips (2-ud, 3-lr)
    #     y = []  # outputs
    #     for si, fi in zip(s, f):
    #         xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
    #         yi = super().predict(xi)[0]  # forward
    #         yi = self._descale_pred(yi, fi, si, img_size)
    #         y.append(yi)
    #     y = self._clip_augmented(y)  # clip augmented tails
    #     return torch.cat(y, -1), None  # augmented inference, train
    
def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    import ast

    # Args
    legacy = True  # backward compatibility for v3/v5/v8/v9 models
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape, num_extra_parameters = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape", "num_extra_parameters"))
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                try:
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
                except ValueError:
                    pass
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in {
            Classify,
            Conv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            C2fPSA,
            C2PSA,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            RepNCSPELAN4,
            ELAN1,
            ADown,
            AConv,
            SPPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
            PSA,
            SCDown,
            C2fCIB,
        }:
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)  # embed channels
                args[2] = int(
                    max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2]
                )  # num heads

            args = [c1, c2, *args[1:]]
            if m in {
                BottleneckCSP,
                C1,
                C2,
                C2f,
                C3k2,
                C2fAttn,
                C3,
                C3TR,
                C3Ghost,
                C3x,
                RepC3,
                C2fPSA,
                C2fCIB,
                C2PSA,
            }:
                args.insert(2, n)  # number of repeats
                n = 1
            if m is C3k2:  # for M/L/X sizes
                legacy = False
                if scale in "mlx":
                    args[3] = True
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in {HGStem, HGBlock}:
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in {Detect, WorldDetect, Segment, Pose, OBB,ZAxis, ImagePoolingAttn, v10Detect}:
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            if m in {Detect, Segment, Pose, OBB,ZAxis}:
                m.legacy = legacy
        elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
            args.insert(1, [ch[x] for x in f])
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        else:
            # print(f)
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m_.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)