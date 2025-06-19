from ultralytics.engine import exporter as ul_exporter
from ultralytics.engine.exporter import export_formats, select_device, colorstr, LOGGER, file_size, try_export
from ultralytics.utils import  __version__, LINUX, DEFAULT_CFG
from ultralytics.utils.checks import check_imgsz
import time 
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from ultralytics.utils import (
    ARM64,
    DEFAULT_CFG,
    IS_JETSON,
    LINUX,
    LOGGER,
    MACOS,
    PYTHON_VERSION,
    ROOT,
    WINDOWS,
    __version__,
    callbacks,
    colorstr,
    get_default_args,
    yaml_save,
)
import torch
import gc
import json
from ultralytics.cfg import TASK2DATA
from ultralytics.nn.autobackend import check_class_names, default_class_names
from ultralytics.nn.modules import C2f, Classify, Detect, RTDETRDecoder
from ultralytics.nn.tasks import WorldModel
from ultralytics.utils.checks import check_imgsz, check_is_path_safe, check_requirements, check_version

class Exporter(ul_exporter.Exporter):
    def __call__(self, model=None) -> str:
        """Returns list of exported files/dirs after running callbacks."""
        self.run_callbacks("on_export_start")
        t = time.time()
        fmt = self.args.format.lower()  # to lowercase
        if fmt in {"tensorrt", "trt"}:  # 'engine' aliases
            fmt = "engine"
        if fmt in {"mlmodel", "mlpackage", "mlprogram", "apple", "ios", "coreml"}:  # 'coreml' aliases
            fmt = "coreml"
        fmts = tuple(export_formats()["Argument"][1:])  # available export formats
        if fmt not in fmts:
            import difflib

            # Get the closest match if format is invalid
            matches = difflib.get_close_matches(fmt, fmts, n=1, cutoff=0.6)  # 60% similarity required to match
            if not matches:
                raise ValueError(f"Invalid export format='{fmt}'. Valid formats are {fmts}")
            LOGGER.warning(f"WARNING ⚠️ Invalid export format='{fmt}', updating to format='{matches[0]}'")
            fmt = matches[0]
        flags = [x == fmt for x in fmts]
        if sum(flags) != 1:
            raise ValueError(f"Invalid export format='{fmt}'. Valid formats are {fmts}")
        (jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, mnn, ncnn, imx, rknn) = (
            flags  # export booleans
        )
        is_tf_format = any((saved_model, pb, tflite, edgetpu, tfjs))

        # Device
        dla = None
        if fmt == "engine" and self.args.device is None:
            LOGGER.warning("WARNING ⚠️ TensorRT requires GPU export, automatically assigning device=0")
            self.args.device = "0"
        if fmt == "engine" and "dla" in str(self.args.device):  # convert int/list to str first
            dla = self.args.device.split(":")[-1]
            self.args.device = "0"  # update device to "0"
            assert dla in {"0", "1"}, f"Expected self.args.device='dla:0' or 'dla:1, but got {self.args.device}."
        self.device = select_device("cpu" if self.args.device is None else self.args.device)

        # Checks
        if imx and not self.args.int8:
            LOGGER.warning("WARNING ⚠️ IMX only supports int8 export, setting int8=True.")
            self.args.int8 = True
        if not hasattr(model, "names"):
            model.names = default_class_names()
        model.names = check_class_names(model.names)
        if self.args.half and self.args.int8:
            LOGGER.warning("WARNING ⚠️ half=True and int8=True are mutually exclusive, setting half=False.")
            self.args.half = False
        if self.args.half and onnx and self.device.type == "cpu":
            LOGGER.warning("WARNING ⚠️ half=True only compatible with GPU export, i.e. use device=0")
            self.args.half = False
            assert not self.args.dynamic, "half=True not compatible with dynamic=True, i.e. use only one."
        self.imgsz = check_imgsz(self.args.imgsz, stride=model.stride, min_dim=2)  # check image size
        if self.args.int8 and engine:
            self.args.dynamic = True  # enforce dynamic to export TensorRT INT8
        if self.args.optimize:
            assert not ncnn, "optimize=True not compatible with format='ncnn', i.e. use optimize=False"
            assert self.device.type == "cpu", "optimize=True not compatible with cuda devices, i.e. use device='cpu'"
        if self.args.int8 and tflite:
            assert not getattr(model, "end2end", False), "TFLite INT8 export not supported for end2end models."
        if edgetpu:
            if not LINUX:
                raise SystemError("Edge TPU export only supported on Linux. See https://coral.ai/docs/edgetpu/compiler")
            elif self.args.batch != 1:  # see github.com/ultralytics/ultralytics/pull/13420
                LOGGER.warning("WARNING ⚠️ Edge TPU export requires batch size 1, setting batch=1.")
                self.args.batch = 1
        if isinstance(model, WorldModel):
            LOGGER.warning(
                "WARNING ⚠️ YOLOWorld (original version) export is not supported to any format.\n"
                "WARNING ⚠️ YOLOWorldv2 models (i.e. 'yolov8s-worldv2.pt') only support export to "
                "(torchscript, onnx, openvino, engine, coreml) formats. "
                "See https://docs.ultralytics.com/models/yolo-world for details."
            )
        if self.args.int8 and not self.args.data:
            self.args.data = DEFAULT_CFG.data or TASK2DATA[getattr(model, "task", "detect")]  # assign default data
            LOGGER.warning(
                "WARNING ⚠️ INT8 export requires a missing 'data' arg for calibration. "
                f"Using default 'data={self.args.data}'."
            )

        # Input
        im = torch.zeros(self.args.batch, 1, *self.imgsz).to(self.device)
        file = Path(
            getattr(model, "pt_path", None) or getattr(model, "yaml_file", None) or model.yaml.get("yaml_file", "")
        )
        if file.suffix in {".yaml", ".yml"}:
            file = Path(file.name)

        # Update model
        model = deepcopy(model).to(self.device)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.float()
        model = model.fuse()

        if imx:
            from ultralytics.utils.torch_utils import FXModel

            model = FXModel(model)
        for m in model.modules():
            if isinstance(m, Classify):
                m.export = True
            if isinstance(m, (Detect, RTDETRDecoder)):  # includes all Detect subclasses like Segment, Pose, OBB
                m.dynamic = self.args.dynamic
                m.export = True
                m.format = self.args.format
                m.max_det = self.args.max_det
            elif isinstance(m, C2f) and not is_tf_format:
                # EdgeTPU does not support FlexSplitV while split provides cleaner ONNX graph
                m.forward = m.forward_split
            if isinstance(m, Detect) and imx:
                from ultralytics.utils.tal import make_anchors

                m.anchors, m.strides = (
                    x.transpose(0, 1)
                    for x in make_anchors(
                        torch.cat([s / m.stride.unsqueeze(-1) for s in self.imgsz], dim=1), m.stride, 0.5
                    )
                )

        y = None
        for _ in range(2):
            y = model(im)  # dry runs
        if self.args.half and onnx and self.device.type != "cpu":
            im, model = im.half(), model.half()  # to FP16

        # Filter warnings
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)  # suppress TracerWarning
        warnings.filterwarnings("ignore", category=UserWarning)  # suppress shape prim::Constant missing ONNX warning
        warnings.filterwarnings("ignore", category=DeprecationWarning)  # suppress CoreML np.bool deprecation warning

        # Assign
        self.im = im
        self.model = model
        self.file = file
        self.output_shape = (
            tuple(y.shape)
            if isinstance(y, torch.Tensor)
            else tuple(tuple(x.shape if isinstance(x, torch.Tensor) else []) for x in y)
        )
        self.pretty_name = Path(self.model.yaml.get("yaml_file", self.file)).stem.replace("yolo", "YOLO")
        data = model.args["data"] if hasattr(model, "args") and isinstance(model.args, dict) else ""
        description = f'Ultralytics {self.pretty_name} model {f"trained on {data}" if data else ""}'
        self.metadata = {
            "description": description,
            "author": "Ultralytics",
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 License (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
            "stride": int(max(model.stride)),
            "task": model.task,
            "batch": self.args.batch,
            "imgsz": self.imgsz,
            "names": model.names,
        }  # model metadata
        if model.task == "pose":
            self.metadata["kpt_shape"] = model.model[-1].kpt_shape

        LOGGER.info(
            f"\n{colorstr('PyTorch:')} starting from '{file}' with input shape {tuple(im.shape)} BCHW and "
            f'output shape(s) {self.output_shape} ({file_size(file):.1f} MB)'
        )

        # Exports
        f = [""] * len(fmts)  # exported filenames
        if jit or ncnn:  # TorchScript
            f[0], _ = self.export_torchscript()
        if engine:  # TensorRT required before ONNX
            f[1], _ = self.export_engine(dla=dla)
        if onnx:  # ONNX
            f[2], _ = self.export_onnx()
        if xml:  # OpenVINO
            f[3], _ = self.export_openvino()
        if coreml:  # CoreML
            f[4], _ = self.export_coreml()
        if is_tf_format:  # TensorFlow formats
            self.args.int8 |= edgetpu
            f[5], keras_model = self.export_saved_model()
            if pb or tfjs:  # pb prerequisite to tfjs
                f[6], _ = self.export_pb(keras_model=keras_model)
            if tflite:
                f[7], _ = self.export_tflite(keras_model=keras_model, nms=False, agnostic_nms=self.args.agnostic_nms)
            if edgetpu:
                f[8], _ = self.export_edgetpu(tflite_model=Path(f[5]) / f"{self.file.stem}_full_integer_quant.tflite")
            if tfjs:
                f[9], _ = self.export_tfjs()
        if paddle:  # PaddlePaddle
            f[10], _ = self.export_paddle()
        if mnn:  # MNN
            f[11], _ = self.export_mnn()
        if ncnn:  # NCNN
            f[12], _ = self.export_ncnn()
        if imx:
            f[13], _ = self.export_imx()

        # Finish
        f = [str(x) for x in f if x]  # filter out '' and None
        if any(f):
            f = str(Path(f[-1]))
            square = self.imgsz[0] == self.imgsz[1]
            s = (
                ""
                if square
                else f"WARNING ⚠️ non-PyTorch val requires square images, 'imgsz={self.imgsz}' will not "
                f"work. Use export 'imgsz={max(self.imgsz)}' if val is required."
            )
            imgsz = self.imgsz[0] if square else str(self.imgsz)[1:-1].replace(" ", "")
            predict_data = f"data={data}" if model.task == "segment" and fmt == "pb" else ""
            q = "int8" if self.args.int8 else "half" if self.args.half else ""  # quantization
            LOGGER.info(
                f'\nExport complete ({time.time() - t:.1f}s)'
                f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
                f'\nPredict:         yolo predict task={model.task} model={f} imgsz={imgsz} {q} {predict_data}'
                f'\nValidate:        yolo val task={model.task} model={f} imgsz={imgsz} data={data} {q} {s}'
                f'\nVisualize:       https://netron.app'
            )

        self.run_callbacks("on_export_end")
        return f  # return list of exported files/dirs
    @try_export
    def export_engine(self, dla=None, prefix=colorstr("TensorRT:")):
        """YOLO TensorRT export https://developer.nvidia.com/tensorrt."""
        assert self.im.device.type != "cpu", "export running on CPU but must be on GPU, i.e. use 'device=0'"
        f_onnx, _ = self.export_onnx()  # run before TRT import https://github.com/ultralytics/ultralytics/issues/7016

        try:
            import tensorrt as trt  # noqa
        except ImportError:
            if LINUX:
                check_requirements("tensorrt>7.0.0,!=10.1.0")
            import tensorrt as trt  # noqa
        check_version(trt.__version__, ">=7.0.0", hard=True)
        check_version(trt.__version__, "!=10.1.0", msg="https://github.com/ultralytics/ultralytics/pull/14239")

        # Setup and checks
        LOGGER.info(f"\n{prefix} starting export with TensorRT {trt.__version__}...")
        is_trt10 = int(trt.__version__.split(".")[0]) >= 10  # is TensorRT >= 10
        assert Path(f_onnx).exists(), f"failed to export ONNX file: {f_onnx}"
        f = self.file.with_suffix(".engine")  # TensorRT engine file
        logger = trt.Logger(trt.Logger.INFO)
        if self.args.verbose:
            logger.min_severity = trt.Logger.Severity.VERBOSE

        # Engine builder
        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        workspace = int(self.args.workspace * (1 << 30)) if self.args.workspace is not None else 0
        if is_trt10 and workspace > 0:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
        elif workspace > 0 and not is_trt10:  # TensorRT versions 7, 8
            config.max_workspace_size = workspace
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)
        half = builder.platform_has_fast_fp16 and self.args.half
        int8 = builder.platform_has_fast_int8 and self.args.int8

        # Optionally switch to DLA if enabled
        if dla is not None:
            if not IS_JETSON:
                raise ValueError("DLA is only available on NVIDIA Jetson devices")
            LOGGER.info(f"{prefix} enabling DLA on core {dla}...")
            if not self.args.half and not self.args.int8:
                raise ValueError(
                    "DLA requires either 'half=True' (FP16) or 'int8=True' (INT8) to be enabled. Please enable one of them and try again."
                )
            config.default_device_type = trt.DeviceType.DLA
            config.DLA_core = int(dla)
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

        # Read ONNX file
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(f_onnx):
            raise RuntimeError(f"failed to load ONNX file: {f_onnx}")

        # Network inputs
        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        for inp in inputs:
            LOGGER.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
        for out in outputs:
            LOGGER.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')

        if self.args.dynamic:
            shape = self.im.shape
            if shape[0] <= 1:
                LOGGER.warning(f"{prefix} WARNING ⚠️ 'dynamic=True' model requires max batch size, i.e. 'batch=16'")
            profile = builder.create_optimization_profile()
            min_shape = (1, shape[1], 32, 32)  # minimum input shape
            max_shape = (*shape[:2], *(int(max(1, workspace) * d) for d in shape[2:]))  # max input shape
            for inp in inputs:
                profile.set_shape(inp.name, min=min_shape, opt=shape, max=max_shape)
            config.add_optimization_profile(profile)

        LOGGER.info(f"{prefix} building {'INT8' if int8 else 'FP' + ('16' if half else '32')} engine as {f}")
        if int8:
            config.set_flag(trt.BuilderFlag.INT8)
            config.set_calibration_profile(profile)
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

            class EngineCalibrator(trt.IInt8Calibrator):
                def __init__(
                    self,
                    dataset,  # ultralytics.data.build.InfiniteDataLoader
                    batch: int,
                    cache: str = "",
                ) -> None:
                    trt.IInt8Calibrator.__init__(self)
                    self.dataset = dataset
                    self.data_iter = iter(dataset)
                    self.algo = trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
                    self.batch = batch
                    self.cache = Path(cache)

                def get_algorithm(self) -> trt.CalibrationAlgoType:
                    """Get the calibration algorithm to use."""
                    return self.algo

                def get_batch_size(self) -> int:
                    """Get the batch size to use for calibration."""
                    return self.batch or 1

                def get_batch(self, names) -> list:
                    """Get the next batch to use for calibration, as a list of device memory pointers."""
                    try:
                        im0s = next(self.data_iter)["img"] / 255.0
                        im0s = im0s.to("cuda") if im0s.device.type == "cpu" else im0s
                        return [int(im0s.data_ptr())]
                    except StopIteration:
                        # Return [] or None, signal to TensorRT there is no calibration data remaining
                        return None

                def read_calibration_cache(self) -> bytes:
                    """Use existing cache instead of calibrating again, otherwise, implicitly return None."""
                    if self.cache.exists() and self.cache.suffix == ".cache":
                        return self.cache.read_bytes()

                def write_calibration_cache(self, cache) -> None:
                    """Write calibration cache to disk."""
                    _ = self.cache.write_bytes(cache)

            # Load dataset w/ builder (for batching) and calibrate
            config.int8_calibrator = EngineCalibrator(
                dataset=self.get_int8_calibration_dataloader(prefix),
                batch=2 * self.args.batch,  # TensorRT INT8 calibration should use 2x batch size
                cache=str(self.file.with_suffix(".cache")),
            )

        elif half:
            config.set_flag(trt.BuilderFlag.FP16)

        # Free CUDA memory
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

        # Write file
        build = builder.build_serialized_network if is_trt10 else builder.build_engine
        with build(network, config) as engine, open(f, "wb") as t:
            # # Metadata
            # meta = json.dumps(self.metadata)
            # t.write(len(meta).to_bytes(4, byteorder="little", signed=True))
            # t.write(meta.encode())
            # Model
            t.write(engine if is_trt10 else engine.serialize())

        return f, None