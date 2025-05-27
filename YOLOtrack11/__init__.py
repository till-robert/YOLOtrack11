"""
YOLOtrack11 Module

This module extends the YOLO framework to include custom functionality for the "zaxis" task.
It provides a modified model, trainer, validator, and predictors,
as well as some patches to the original YOLO package.


Classes:
    YOLOtrack11: Custom YOLO class for the "zaxis" task.

Modules:
    dataset, loss, model, val, predict, utils, results, instance
"""

__all__ = ["dataset", "loss", "model", "val", "predict", "utils", "results", "instance", "exporter"]



import ultralytics
from ultralytics.utils import yaml_load, SettingsManager, SETTINGS_FILE
from pathlib import Path
from .utils import imread
from .instance import Instances

# Patches to original YOLO package
ROOT = Path(__file__).resolve().parent  # Patch root directory to point to local module folder
ultralytics.utils.checks.ROOT = ROOT

# Set default configuration
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
get_cfg = lambda cfg=DEFAULT_CFG_DICT, overrides=None: get_cfg_old(cfg, overrides)
ultralytics.cfg.get_cfg = get_cfg
ultralytics.engine.validator.get_cfg = get_cfg



# Patch AutoBackend class
autobackend_base = ultralytics.nn.autobackend.AutoBackend
class AutoBackend(autobackend_base):
    """
    Custom AutoBackend class to modify the warmup behavior for the "zaxis" task.
    """
    def warmup(self, imgsz=(1, 3, 640, 640)):
        """
        Adjusts the warmup image size for the "zaxis" task.

        Args:
            imgsz (tuple): Image size in the format (batch_size, channels, height, width).
        """
        imgsz = list(imgsz)
        imgsz[1] = 1  # Set channels to 1 for "zaxis"
        super().warmup(imgsz)

ultralytics.nn.autobackend.AutoBackend = AutoBackend
ultralytics.engine.validator.AutoBackend = AutoBackend
ultralytics.engine.predictor.AutoBackend = AutoBackend

# Patch imread function
ultralytics.utils.patches.imread = imread
ultralytics.data.loaders.imread = imread

# Patch Instances class
ultralytics.utils.instance.Instances = Instances


from .model import ZAxisModel
from .train import ZAxisTrainer
from .val import ZAxisValidator
from .predict import ZAxisPredictor

from ultralytics.models.yolo import YOLO

class YOLOtrack11(YOLO):
    """
    Custom YOLO class for the "zaxis" task.

    This class extends the YOLO framework to include custom functionality for the "zaxis" task,
    such as custom models, trainers, validators, and predictors.
    """
    def __init__(self, model="yolo11n-zaxis.yaml", task="zaxis", verbose=False):
        """
        Initializes the YOLOtrack11 class.

        Args:
            model (str): Path to the model configuration file.
            task (str): Task name (default is "zaxis").
            verbose (bool): Whether to enable verbose logging.
        """
        super().__init__(model, task, verbose)

    @property
    def task_map(self):
        """
        Maps the task name to the corresponding model, trainer, validator, and predictor classes.

        Returns:
            dict: A dictionary mapping task names to their respective classes.
        """
        return {
            "zaxis": {
                "model": ZAxisModel,
                "trainer": ZAxisTrainer,
                "validator": ZAxisValidator,
                "predictor": ZAxisPredictor,
            },
        }

    def val(self, validator=None, **kwargs):
        """
        Runs the validation process for the model.

        Args:
            validator (callable, optional): Custom validator class. If not provided, the default
                validator for the task will be used.
            **kwargs: Additional arguments to override default validation parameters.

        Returns:
            dict: Validation metrics.
        """
        custom = {}  # Method defaults
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}  # Highest priority args on the right

        validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics
    def export(
        self,
        **kwargs,
    ) -> str:
        self._check_is_pytorch_model()
        from .exporter import Exporter

        custom = {
            "imgsz": self.model.args["imgsz"],
            "batch": 1,
            "data": None,
            "device": None,  # reset to avoid multi-GPU errors
            "verbose": False,
        }  # method defaults
        args = {**self.overrides, **custom, **kwargs, "mode": "export"}  # highest priority args on the right
        return Exporter(overrides=args, _callbacks=self.callbacks)(model=self.model)