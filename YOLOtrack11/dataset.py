import torch
import os
import math
import cv2
from itertools import repeat
from ultralytics.utils import LOGGER, instance
from ultralytics.utils.ops import segments2boxes,resample_segments
import glob
from ultralytics.data.loaders import (
    LoadImagesAndVideos as LoadImagesAndVideos_ultralytics,
    LoadPilAndNumpy,
    LoadScreenshots,
    LoadStreams,
    LoadTensor as LoadTensor_ultralytics,
    SourceTypes,
)
from ultralytics.data.build import check_source
from ultralytics.data.dataset import YOLODataset,DATASET_CACHE_VERSION
from ultralytics.data.utils import exif_size,IMG_FORMATS,VID_FORMATS,FORMATS_HELP_MSG,ImageOps,get_hash,HELP_URL,save_dataset_cache_file
import numpy as np
from pathlib import Path
from multiprocessing.pool import ThreadPool
from ultralytics.utils import LOCAL_RANK, NUM_THREADS, TQDM, colorstr
from PIL import Image
from .augment import v8_transforms, Instances, Compose, LetterBox, Format
from .utils import get_n_frames, imread
from ultralytics.utils.checks import check_requirements


class LoadTensor(LoadTensor_ultralytics):    
    @staticmethod
    def _single_check(im, stride=32):
        """Validates and formats a single image tensor, ensuring correct shape and normalization."""
        s = (
            f"WARNING ⚠️ torch.Tensor inputs should be BCHW i.e. shape(1, 3, 640, 640) "
            f"divisible by stride {stride}. Input shape{tuple(im.shape)} is incompatible."
        )
        if len(im.shape) != 4:
            if len(im.shape) != 3:
                raise ValueError(s)
            LOGGER.warning(s)
            im = im.unsqueeze(0)
        if im.shape[2] % stride or im.shape[3] % stride:
            raise ValueError(s)
        if im.max() > 1.0 + torch.finfo(im.dtype).eps:  # torch.float32 eps is 1.2e-07
            # normalization is done in the preprocessing step (e.g. in val.py)
            im = im.float()

        return im

class LoadImagesAndVideos(LoadImagesAndVideos_ultralytics):
    """Dataloader for images and videos, supporting various input formats.
    This class extends the LoadImagesAndVideos from ultralytics.data.loaders to support multi-frame TIFF files."""

    def __init__(self, path, batch=1, vid_stride=1):
        """Initialize dataloader for images and videos, supporting various input formats."""
        parent = None
        if isinstance(path, str) and Path(path).suffix == ".txt":  # *.txt file with img/vid/dir on each line
            parent = Path(path).parent
            path = Path(path).read_text().splitlines()  # list of sources
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            a = str(Path(p).absolute())  # do not use .resolve() https://github.com/ultralytics/ultralytics/issues/2912
            if "*" in a:
                files.extend(sorted(glob.glob(a, recursive=True)))  # glob
            elif os.path.isdir(a):
                files.extend(sorted(glob.glob(os.path.join(a, "*.*"))))  # dir
            elif os.path.isfile(a):
                files.append(a)  # files (absolute or relative to CWD)
            elif parent and (parent / p).is_file():
                files.append(str((parent / p).absolute()))  # files (relative to *.txt file parent)
            else:
                raise FileNotFoundError(f"{p} does not exist")

        # Define files as images or videos
        images, videos, multitifs = [], [], []
        for f in files:
            suffix = f.split(".")[-1].lower()  # Get file extension without the dot and lowercase
            if suffix in {"tif", "tiff"} and get_n_frames(f) > 1:  # TIFF files
                multitifs.append(f)  # treat multi-frame TIFF as video
            elif suffix in IMG_FORMATS:
                images.append(f)
            elif suffix in VID_FORMATS:
                videos.append(f)
        ni, nv, nt = len(images), len(videos), len(multitifs)

        self.files = images + videos + multitifs
        self.nf = ni + nv + nt  # number of files
        self.ni = ni  # number of images
        self.video_flag = [False] * ni + [True] * nv + [False] * nt  # video flag for each file
        self.tif_flag = [False] * ni + [False] * nv + [True] * nt  # tif flag for each file
        self.mode = "video" if ni == 0 else "image"  # default to video if no images
        self.vid_stride = vid_stride  # video frame-rate stride
        self.bs = batch
        if any(videos):
            self._new_video(videos[0])  # new video
        if any(multitifs):
            self._new_multitif(multitifs[0])
        else:
            self.cap = None
        if self.nf == 0:
            raise FileNotFoundError(f"No images or videos found in {p}. {FORMATS_HELP_MSG}")
    
    def _new_multitif(self, path):
        """Creates a new video capture object for the given path and initializes video-related attributes."""
        self.frame = 0
        self.multitif = Image.open(path)
        self.fps = 25
        self.frames = int(self.multitif.n_frames / self.vid_stride)
    
    def __next__(self):
        """Returns the next batch of images or video frames with their paths and metadata."""
        paths, imgs, info = [], [], []
        while len(imgs) < self.bs:
            if self.count >= self.nf:  # end of file list
                if imgs:
                    return paths, imgs, info  # return last partial batch
                else:
                    raise StopIteration

            path = self.files[self.count]
            if self.video_flag[self.count]:
                self.mode = "video"
                if not self.cap or not self.cap.isOpened():
                    self._new_video(path)

                success = False
                for _ in range(self.vid_stride):
                    success = self.cap.grab()
                    if not success:
                        break  # end of video or failure

                if success:
                    success, im0 = self.cap.retrieve()
                    if success:
                        self.frame += 1
                        paths.append(path)
                        imgs.append(im0)
                        info.append(f"video {self.count + 1}/{self.nf} (frame {self.frame}/{self.frames}) {path}: ")
                        if self.frame == self.frames:  # end of video
                            self.count += 1
                            self.cap.release()
                else:
                    # Move to the next file if the current video ended or failed to open
                    self.count += 1
                    if self.cap:
                        self.cap.release()
                    if self.count < self.nf:
                        self._new_video(self.files[self.count])
            if self.tif_flag[self.count]:
                self.mode = "video"
                if not self.multitif:
                    self._new_multitif(path)

                success = False
                for _ in range(self.vid_stride):
                    try:
                        self.multitif.seek(self.frame * self.vid_stride)  # seek to the next frame
                        success = True
                    except EOFError:
                        success = False
                    if not success:
                        break  # end of video or failure

                if success:
                    im0 = np.asarray(self.multitif)
                    self.frame += 1
                    paths.append(path)
                    imgs.append(im0)
                    info.append(f"video {self.count + 1}/{self.nf} (frame {self.frame}/{self.frames}) {path}: ")
                    if self.frame == self.frames:  # end of video
                        self.count += 1
                        self.multitif.close()
                        self.multitif = None
                else:
                    # Move to the next file if the current video ended or failed to open
                    self.count += 1
                    if self.multitif:
                        self.multitif.close()
                        self.multitif = None
                    # Check if there are more files to process
                    if self.count < self.nf:
                        self._new_multitif(self.files[self.count])
            else:
                # Handle image files (including HEIC)
                self.mode = "image"
                if path.split(".")[-1].lower() == "heic":
                    # Load HEIC image using Pillow with pillow-heif
                    check_requirements("pillow-heif")

                    from pillow_heif import register_heif_opener

                    register_heif_opener()  # Register HEIF opener with Pillow
                    with Image.open(path) as img:
                        im0 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)  # convert image to BGR nparray
                else:
                    im0 = imread(path)  # BGR
                if im0 is None:
                    LOGGER.warning(f"WARNING ⚠️ Image Read Error {path}")
                else:
                    paths.append(path)
                    imgs.append(im0)
                    info.append(f"image {self.count + 1}/{self.nf} {path}: ")
                self.count += 1  # move to the next file
                if self.count >= self.ni:  # end of image list
                    break

        return paths, imgs, info


def load_inference_source(source=None, batch=1, vid_stride=1, buffer=False):
    """
    Loads an inference source for object detection and applies necessary transformations.

    Args:
        source (str, Path, Tensor, PIL.Image, np.ndarray): The input source for inference.
        batch (int, optional): Batch size for dataloaders. Default is 1.
        vid_stride (int, optional): The frame interval for video sources. Default is 1.
        buffer (bool, optional): Determined whether stream frames will be buffered. Default is False.

    Returns:
        dataset (Dataset): A dataset object for the specified input source.
    """
    source, stream, screenshot, from_img, in_memory, tensor = check_source(source)
    source_type = source.source_type if in_memory else SourceTypes(stream, screenshot, from_img, tensor)

    # Dataloader
    if tensor:
        dataset = LoadTensor(source)
    elif in_memory:
        dataset = source
    elif stream:
        dataset = LoadStreams(source, vid_stride=vid_stride, buffer=buffer)
    elif screenshot:
        dataset = LoadScreenshots(source)
    elif from_img:
        dataset = LoadPilAndNumpy(source)
    else:
        dataset = LoadImagesAndVideos(source, batch=batch, vid_stride=vid_stride)

    # Attach source types to the dataset
    setattr(dataset, "source_type", source_type)

    return dataset
def verify_image_label(args):
    """Verify one image-label pair."""
    im_file, lb_file, prefix, keypoint,zaxis, num_cls, nkpt, ndim, nparams = args
    #print(args)
    # Number (missing, found, empty, corrupt), message, segments, keypoints
    nm, nf, ne, nc, msg, segments, keypoints, extra_parameters = 0, 0, 0, 0, "", [], None, None
# try:
    # Verify images
    im = Image.open(im_file)
    im.verify()  # PIL verify
    shape = exif_size(im)  # image size
    shape = (shape[1], shape[0])  # hw
    assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
    assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}. {FORMATS_HELP_MSG}"
    if im.format.lower() in {"jpg", "jpeg"}:
        with open(im_file, "rb") as f:
            f.seek(-2, 2)
            if f.read() != b"\xff\xd9":  # corrupt JPEG
                ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                msg = f"{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"

    # Verify labels
    if os.path.isfile(lb_file):
        nf = 1  # label found
        with open(lb_file) as f:
            lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
            if any(len(x) > 6 for x in lb) and (not keypoint and not zaxis):  # is segment
                classes = np.array([x[0] for x in lb], dtype=np.float32)
                segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
            lb = np.array(lb, dtype=np.float32)
        nl = len(lb)
        if nl:
            if keypoint:
                assert lb.shape[1] == (5 + nkpt * ndim), f"labels require {(5 + nkpt * ndim)} columns each"
                points = lb[:, 5:].reshape(-1, ndim)[:, :2]
                assert points.max() <= 1, f"non-normalized or out of bounds coordinates {points[points > 1]}"
            elif zaxis:
                assert lb.shape[1] == (5 + nparams + nkpt*ndim), f"zaxis labels require {(5 + nparams + nkpt * ndim)} columns each "
                extra_parameters = lb[:,5:5+nparams] #includes z, orientation,...
                points = lb[:, 5+nparams:].reshape(-1, ndim)[:, :2]
                # assert points.max() <= 1, f"non-normalized or out of bounds coordinates {points[points > 1]}"

            else:
                assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
                points = lb[:, 1:]
                # assert z_positions.max() <= 1, f"non-normalized or out of bounds coordinates {z_positions[z_positions > 1]}"
            # assert lb.min() >= 0, f"negative label values {lb[lb < 0]}"

            # All labels
            max_cls = lb[:, 0].max()  # max label count
            assert max_cls <= num_cls, (
                f"Label class {int(max_cls)} exceeds dataset class count {num_cls}. "
                f"Possible class labels are 0-{num_cls - 1}"
            )
            _, i = np.unique(lb, axis=0, return_index=True)
            if len(i) < nl:  # duplicate row check
                lb = lb[i]  # remove duplicates
                if segments:
                    segments = [segments[x] for x in i]
                msg = f"{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed"
        else:
            ne = 1  # label empty
            lb = np.zeros((0, (5 + nkpt * ndim+nparams) if zaxis else 5), dtype=np.float32)
    else:
        nm = 1  # label missing
        lb = np.zeros((0, (5 + nkpt * ndim+nparams) if zaxis else 5), dtype=np.float32)
    if keypoint:
        keypoints = lb[:, 5:].reshape(-1, nkpt, ndim)
        if ndim == 2:
            kpt_mask = np.where((keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0).astype(np.float32)
            keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)  # (nl, nkpt, 3)
    elif zaxis:

        keypoints = lb[:, 5+nparams:].reshape(-1, nkpt, ndim)
        extra_parameters = lb[:, 5:5+nparams]
        if ndim == 2:
            kpt_mask = np.where((keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0).astype(np.float32)
            keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)  # (nl, nkpt, 3)
    lb = lb[:, :5]

    return im_file, lb, shape, segments, keypoints,extra_parameters, nm, nf, ne, nc, msg

class YOLOtrackDataset(YOLODataset):


    def __init__(self, *args, task="zaxis", **kwargs):
        self.use_zaxis = task == "zaxis"
        #print(task,data)
        super().__init__(*args, **kwargs)

    def cache_labels(self, path=Path("./labels.cache")):
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): Path where to save the cache file. Default is Path('./labels.cache').

        Returns:
            (dict): labels.
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        n_extra_parameters = self.data.get("num_extra_parameters", 1)
        if (self.use_keypoints or self.use_zaxis) and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(self.use_zaxis),
                    repeat(len(self.data["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                    repeat(n_extra_parameters),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, segments, keypoint,extra_parameters, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": lb[:, 0:1],  # n, 1
                            "bboxes": lb[:, 1:],  # n, 4
                            "segments": segments,
                            "keypoints": keypoint,
                            "extra_parameters": extra_parameters,
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x


    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False, background_value=hyp.background)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                return_zaxis=self.use_zaxis,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )
        return transforms
        return None

    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label):

        bboxes = label.pop("bboxes")
        extra_parameters = label.pop("extra_parameters",[])
        if(extra_parameters is None):
            extra_parameters = np.empty(0)
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        # NOTE: do NOT resample oriented boxes
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # list[np.array(1000, 2)] * num_samples
            # (N, 1000, 2)
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        
        label["instances"] = Instances(bboxes, segments, keypoints,extra_parameters, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch):
        try:
            """Collates data samples into batches."""
            new_batch = {}
            keys = batch[0].keys()
            values = list(zip(*[list(b.values()) for b in batch]))
            for i, k in enumerate(keys):
                value = values[i]
                if k == "img":
                    value = torch.stack(value, 0).contiguous()
                if k in {"masks", "keypoints", "bboxes", "cls","extra_parameters", "segments", "obb"}:
                    value = torch.cat(value, 0).contiguous()
                new_batch[k] = value
            new_batch["batch_idx"] = list(new_batch["batch_idx"])
            for i in range(len(new_batch["batch_idx"])):
                new_batch["batch_idx"][i] += i  # add target image index for build_targets()
            new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0).contiguous()
            return new_batch
        except Exception as e:
            print("Collate failed on batch:", batch)
            raise e
    def load_image(self, i, rect_mode=True):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                try:
                    im = np.load(fn)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}WARNING ⚠️ Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    im = cv2.imread(f,cv2.IMREAD_UNCHANGED)  # BGR
                    if(len(im.shape)==2): im = im[None,...]
            else:  # read image
                im = cv2.imread(f,cv2.IMREAD_UNCHANGED)  # BGR
                if len(im.shape)==2 : im[None,...]#if greyscale
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")

            h0, w0 = im.shape[:2]  # orig hw
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                r = self.imgsz / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]
    def update_labels(self, include_class):
        """Update labels to include only these classes (optional)."""
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                segments = self.labels[i]["segments"]
                keypoints = self.labels[i]["keypoints"]
                extra_parameters = self.labels[i]["extra_parameters"]
                j = (cls == include_class_array).any(1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]
                if segments:
                    self.labels[i]["segments"] = [segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    self.labels[i]["keypoints"] = keypoints[j]
                if extra_parameters is not None:
                    self.labels[i]["extra_parameters"] = extra_parameters[j]
            if self.single_cls:
                self.labels[i]["cls"][:, 0] = 0