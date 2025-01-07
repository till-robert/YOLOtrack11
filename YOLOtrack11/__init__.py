__all__ = ["dataset", "loss", "model","val","predict","utils","results", "instance"]


import ultralytics
from ultralytics.utils import yaml_load
from pathlib import Path
from .utils import imread
from .instance import Instances

## Patches to original YOLO package

ROOT = Path(__file__).resolve().parent #Patch root directory to point to local module folder
ultralytics.utils.checks.ROOT = ROOT
# set Default configuration
DEFAULT_CFG_PATH = ROOT / "default.yaml"
DEFAULT_CFG_DICT = yaml_load(DEFAULT_CFG_PATH)
for k, v in DEFAULT_CFG_DICT.items():
    if isinstance(v, str) and v.lower() == "none":
        DEFAULT_CFG_DICT[k] = None
ultralytics.utils.DEFAULT_CFG_DICT = DEFAULT_CFG_DICT
ultralytics.utils.DEFAULT_CFG_KEYS = DEFAULT_CFG_DICT.keys()
ultralytics.utils.DEFAULT_CFG = ultralytics.utils.IterableSimpleNamespace(**DEFAULT_CFG_DICT)
ultralytics.cfg.TASK2DATA["zaxis"] = ""
get_cfg_old = ultralytics.cfg.get_cfg
get_cfg = lambda cfg = DEFAULT_CFG_DICT, overrides= None: get_cfg_old(cfg, overrides)
ultralytics.cfg.get_cfg = get_cfg
ultralytics.engine.validator.get_cfg = get_cfg


autobackend_base = ultralytics.nn.autobackend.AutoBackend
class AutoBackend(autobackend_base):
    def warmup(self, imgsz=(1, 3, 640, 640)):
        imgsz = list(imgsz)
        imgsz[1] = 1
        super().warmup(imgsz)
ultralytics.nn.autobackend.AutoBackend = AutoBackend
ultralytics.engine.validator.AutoBackend = AutoBackend
ultralytics.engine.predictor.AutoBackend = AutoBackend
ultralytics.utils.patches.imread = imread
ultralytics.data.loaders.imread = imread

ultralytics.utils.instance.Instances = Instances



from .model import ZAxisModel
from .train import ZAxisTrainer
from .val import ZAxisValidator
from .predict import ZAxisPredictor

from ultralytics.models.yolo import YOLO

class YOLOtrack11(YOLO):
    def __init__(self, model="yolo11n-zaxis.yaml", task="zaxis", verbose=False):
        super().__init__(model, task, verbose)
    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "zaxis": {
                "model": ZAxisModel,
                "trainer": ZAxisTrainer,
                "validator": ZAxisValidator,
                "predictor": ZAxisPredictor,
            },
        }
    
    def val(
        self,
        validator=None,
        **kwargs,
    ):

        custom = {}  # method defaults
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}  # highest priority args on the right

        validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics