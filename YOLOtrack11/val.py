        # Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path
import json

import torch

from ultralytics.models.yolo.pose import PoseValidator
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops, callbacks, emojis,TQDM,colorstr
from ultralytics.utils.metrics import batch_probiou,box_iou,Metric,compute_ap,SimpleClass, plot_mc_curve, plot_pr_curve, smooth
from ultralytics.utils.torch_utils import smart_inference_mode, select_device, de_parallel
from ultralytics.utils.checks import check_imgsz
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.utils.ops import Profile
from ultralytics.data.build import build_dataloader

from .dataset import YOLOtrackDataset
from .plotting import output_to_z_target, plot_images
from .utils import scale_boxes, scale_coords #,scale_to_physical
from ultralytics.nn.autobackend import AutoBackend
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import r2_score,mean_squared_error


class ZAxisValidator(PoseValidator):
    """
    A class extending the Validator class based on an Oriented Bounding Box (OBB) and Pose Validator.

    Example:
        ```python

        args = dict(model="yolov8n-zaxis.yml", data="path/to/dataset.yaml")
        validator = ZAxisValidator(args=args)
        validator(model=args["model"])
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = "zaxis"
        self.metrics = ZAxisMetrics(save_dir=self.save_dir, plot=True, on_plot=self.on_plot)
        self.physical_scale = args["physical_scale"] if "physical_scale" in args else (1,1,1)
        



    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        super().init_metrics(model)
        val = self.data.get(self.args.split, "")  # validation path
        self.is_dota = isinstance(val, str) and "DOTA" in val  # is COCO
        self.ne = self.data["num_extra_parameters"]
        # self.stats["pred_z"] = []
        # self.stats["target_z"] = []
        self.stats["xyz_pairs"] = []
        del self.stats["tp_p"]
        


    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            nc=self.nc,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            rotated=False,
        )

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape (N, 6) representing detections where each detection is
                (x1, y1, x2, y2, conf, class).
            gt_bboxes (torch.Tensor): Tensor of shape (M, 4) representing ground-truth bounding box coordinates. Each
                bounding box is of the format: (x1, y1, x2, y2).
            gt_cls (torch.Tensor): Tensor of shape (M,) representing target class indices.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape (N, 10) for 10 IoU levels.

        Note:
            The function does not return any value directly usable for metrics calculation. Instead, it provides an
            intermediate representation used for evaluating predictions against ground truth.
        """
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou)
    def match_predictions(self, pred_classes, true_classes, iou):
        """
        Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape(N,).
            true_classes (torch.Tensor): Target class indices of shape(M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground of truth
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape(N,10) for 10 IoU thresholds.
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        gt_pred_matches = np.zeros((self.iouv.shape[0],true_classes.shape[0],pred_classes.shape[0] ), dtype=bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()
        # gt_idx = gt_idx.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):

            # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708

            cost_matrix = iou * (iou >= threshold)
            if cost_matrix.any():
                labels_idx, detections_idx = linear_sum_assignment(cost_matrix, maximize=True)
                valid = cost_matrix[labels_idx, detections_idx] > 0
                if valid.any():
                    correct[detections_idx[valid], i] = True
                    gt_pred_matches[i,labels_idx,detections_idx] = valid
            # else:
            #     matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
            #     matches = np.array(matches).T
            #     if matches.shape[0]:
            #         if matches.shape[0] > 1:
            #             matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
            #             matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            #             # matches = matches[matches[:, 2].argsort()[::-1]]
            #             matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            #         correct[matches[:, 1].astype(int), i] = True
            #         matched_z[matches[:, 1].astype(int), i] = true_z[]
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device),torch.tensor(gt_pred_matches,device=pred_classes.device)
    def _prepare_batch(self, si, batch):
        """Prepares and returns a batch for OBB validation."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        z = batch["extra_parameters"][idx][:,0:1]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch.pop("ratio_pad", None)
        ratio_pad = ratio_pad[si] if ratio_pad else ((1,1),(0,0))

        kpts = batch["keypoints"][idx]
        h, w = imgsz
        kpts = kpts.clone()
        kpts[..., 0] *= w
        kpts[..., 1] *= h
        kpts = scale_coords(imgsz, kpts, ori_shape, ratio_pad=ratio_pad)

        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
            bbox = scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)  # native-space labels
        return {"cls": cls, "bbox": bbox, "z": z,"ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad, "img":batch["img"][si],"kpts": kpts}

    def _prepare_pred(self, pred, pbatch):
        """Prepares and scales keypoints in a batch for pose processing."""
        predn = pred.clone()
        predn[:, :4] = scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )  # native-space pred
        nk = pbatch["kpts"].shape[1]
        pred_kpts = predn[:, 6+self.ne:].view(len(predn), nk, -1)
        pred_kpts = scale_coords(pbatch["imgsz"], pred_kpts, pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"])
        return predn, pred_kpts

    def update_metrics(self, preds, batch):
        
        """Metrics."""
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                xyz_pairs=torch.zeros(len(batch["keypoints"]),2,3, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
                # tp_p=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox,gt_z,gt_kpts = pbatch.pop("cls"), pbatch.pop("bbox"),pbatch.pop("z").squeeze(-1), pbatch.get("kpts")[...,:-1] #remove visibility from kpts
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            # stat["target_z"] = z
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn, pred_kpts = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            pred_z = predn[:,6].squeeze(-1)

            # Evaluate
            if nl:
                stat["tp"],gt_pred_matches_all = self._process_batch(predn, bbox, cls)
                # matched = (pred_z*gt_pred_matcher).sum(axis=2) #z predictions matched to ground truth for different iou values, unmatched are 0
                # matched_kpt = (gt_pred_matcher.unsqueeze(-1)*pred_kpts.squeeze(1)).sum(axis=2) #kpt predictions matched to ground truth for different iou values, unmatched are 0
                # matched[~gt_pred_matcher.sum(axis=2).type(torch.bool)] = torch.nan #set unmatched detections to nan
                # matched_kpt[~gt_pred_matcher.sum(axis=2).type(torch.bool)] = torch.nan #set unmatched detections to nan

                pred_xyz = torch.concat([pred_kpts.squeeze(), pred_z[...,None]], dim=-1)
                gt_xyz = torch.concat([gt_kpts.squeeze(), gt_z[...,None]], dim=-1)

                # Get indices where condition is met
                gt_pred_matches = gt_pred_matches_all[6]  # choose iou=0.8
                gt_idx, pred_idx = torch.where(gt_pred_matches)
                matched_xyz_pairs = torch.stack([gt_xyz[gt_idx], pred_xyz[pred_idx]], dim=1)

                # Identify unmatched indices
                all_gt = torch.arange(nl, device=self.device)
                all_pred = torch.arange(npr, device=self.device)

                # Identify unmatched ground truth indices
                matched_gt = torch.unique(gt_idx)
                unmatched_gt_mask = ~torch.isin(all_gt, matched_gt)
                unmatched_gt_xyz = gt_xyz[unmatched_gt_mask]
                unmatched_gt_xyz_pairs = torch.stack([unmatched_gt_xyz, torch.full_like(unmatched_gt_xyz, float('nan'), device=self.device)], dim=1)

                # Identify unmatched prediction indices
                matched_pred = torch.unique(pred_idx)
                unmatched_pred_mask = ~torch.isin(all_pred, matched_pred)
                unmatched_pred_xyz = pred_xyz[unmatched_pred_mask]
                unmatched_pred_xyz_pairs = torch.stack([torch.full_like(unmatched_pred_xyz, float('nan'), device=self.device), unmatched_pred_xyz], dim=1)

                stat["xyz_pairs"] = torch.cat([matched_xyz_pairs, unmatched_gt_xyz_pairs, unmatched_pred_xyz_pairs], dim=0)


                # stat["z_pairs"] = torch.cat([gt_z.expand((len(matched),-1)).unsqueeze(-1),matched.unsqueeze(-1)],2).transpose(0,1) #paired up z-values
                # stat["kpt_pairs"] = torch.cat([gt_kpts.transpose(0,1).expand((len(matched),-1,-1)).unsqueeze(-2),matched_kpt.unsqueeze(-2)],-2).transpose(0,1) #paired up kpt-values
                #stat["kpt_pairs"],stat["z_pairs"] = scale_to_physical(stat["kpt_pairs"],stat["z_pairs"], self.physical_scale, pbatch["ori_shape"])

                #stat["z_pairs"] = [[pair for pair in row if not torch.any(torch.isnan(pair))] for row in z_pairs] #turn into list where nans are excluded
                # stat["tp_z"] = self.
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, bbox, cls)

            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    # pred_z,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f'{Path(batch["im_file"][si]).stem}.txt',
                )
    
    def plot_val_samples(self, batch, ni):
        """Plot validation image samples."""
        plot_images( #TODO: use custom plotting function
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
            z=batch["extra_parameters"].squeeze(-1)

        )


    def save_one_txt(self, predn, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        from ultralytics.engine.results import Results
        import numpy as np
        # zaxis = torch.cat([predn, pred_z.view(-1,1)],1)

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            zaxis=predn
        ).save_txt(file, save_conf=save_conf)

    def preprocess(self, batch):
        """Preprocesses batch of images for YOLO training."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        is_16bit = batch["img"].dtype == torch.uint16
        batch["img"] = (batch["img"].clamp(0,65504.).half() if self.args.half else batch["img"].float()) #unsafe conversion
        if is_16bit:
            batch["img"] = (batch["img"]-self.args.level)/self.args.window
        else:
            batch["img"] /= 255
        for k in ["batch_idx", "cls", "bboxes","extra_parameters","keypoints"]:
            batch[k] = batch[k].to(self.device)

        if self.args.save_hybrid:
            height, width = batch["img"].shape[2:]
            nb = len(batch["img"])
            bboxes = batch["bboxes"] * torch.tensor((width, height, width, height), device=self.device)
            self.lb = [
                torch.cat([batch["cls"][batch["batch_idx"] == i], bboxes[batch["batch_idx"] == i]], dim=-1)
                for i in range(nb)
            ]

        return batch
    def get_desc(self):
        """Return a formatted string summarizing class metrics of YOLO model."""
        return ("%22s" + "%11s" * 8) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)","z rms.","xy rms.")
    
    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        # return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

        print("image size:", self.args.imgsz)
        return YOLOtrackDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,  # augmentation
            hyp=self.args,  # TODO: probably add a get_hyps_from_self.args function
            rect=self.args.rect,  # rectangular batches
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=int(self.stride),
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )
    
    # def get_dataloader(self, dataset_path, batch_size):
    #     """Construct and return dataloader."""
    #     dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
    #     return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)  # return dataloader
    # @smart_inference_mode()
    # def __call__(self, trainer=None, model=None):
    #     """Executes validation process, running inference on dataloader and computing performance metrics."""
    #     self.training = trainer is not None
    #     augment = self.args.augment and (not self.training)
    #     if self.training:
    #         self.device = trainer.device
    #         self.data = trainer.data
    #         # force FP16 val during training
    #         self.args.half = self.device.type != "cpu" and trainer.amp
    #         model = trainer.ema.ema or trainer.model
    #         model = model.half() if self.args.half else model.float()
    #         # self.model = model
    #         self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
    #         self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
    #         model.eval()
    #     else:
    #         if str(self.args.model).endswith(".yaml"):
    #             LOGGER.warning("WARNING ⚠️ validating an untrained model YAML will result in 0 mAP.")
    #         callbacks.add_integration_callbacks(self)
    #         model = AutoBackend(
    #             weights=model or self.args.model,
    #             device=select_device(self.args.device, self.args.batch),
    #             dnn=self.args.dnn,
    #             data=self.args.data,
    #             fp16=self.args.half,
    #         )
    #         # self.model = model
    #         self.device = model.device  # update device
    #         self.args.half = model.fp16  # update half
    #         stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
    #         imgsz = check_imgsz(self.args.imgsz, stride=stride)
    #         if engine:
    #             self.args.batch = model.batch_size
    #         elif not pt and not jit:
    #             self.args.batch = model.metadata.get("batch", 1)  # export.py models default to batch-size 1
    #             LOGGER.info(f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})")

    #         if str(self.args.data).split(".")[-1] in {"yaml", "yml"}:
    #             self.data = check_det_dataset(self.args.data)
    #         elif self.args.task == "classify":
    #             self.data = check_cls_dataset(self.args.data, split=self.args.split)
    #         else:
    #             raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

    #         if self.device.type in {"cpu", "mps"}:
    #             self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
    #         if not pt:
    #             self.args.rect = False
    #         self.stride = model.stride  # used in get_dataloader() for padding
    #         self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

    #         model.eval()
    #         model.warmup(imgsz=(1 if pt else self.args.batch, 1 if self.args.task == "zaxis" else 3, imgsz, imgsz))  # warmup

    #     self.run_callbacks("on_val_start")
    #     dt = (
    #         Profile(device=self.device),
    #         Profile(device=self.device),
    #         Profile(device=self.device),
    #         Profile(device=self.device),
    #     )
    #     bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
    #     self.init_metrics(de_parallel(model))
    #     self.jdict = []  # empty before each val
    #     for batch_i, batch in enumerate(bar):
    #         self.run_callbacks("on_val_batch_start")
    #         self.batch_i = batch_i
    #         # Preprocess
    #         with dt[0]:
    #             batch = self.preprocess(batch)

    #         # Inference
    #         with dt[1]:
    #             preds = model(batch["img"], augment=augment)

    #         # Loss
    #         with dt[2]:
    #             if self.training:
    #                 self.loss += model.loss(batch, preds)[1]

    #         # Postprocess
    #         with dt[3]:
    #             preds = self.postprocess(preds)

    #         self.update_metrics(preds, batch)
    #         if self.args.plots and batch_i < 3:
    #             self.plot_val_samples(batch, batch_i)
    #             self.plot_predictions(batch, preds, batch_i)

    #         self.run_callbacks("on_val_batch_end")
    #     stats = self.get_stats()
    #     self.check_stats(stats)
    #     self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
    #     self.finalize_metrics()
    #     self.print_results()
    #     self.run_callbacks("on_val_end")
    #     if self.training:
    #         model.float()
    #         results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
    #         return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
    #     else:
    #         LOGGER.info(
    #             "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
    #                 *tuple(self.speed.values())
    #             )
    #         )
    #         if self.args.save_json and self.jdict:
    #             with open(str(self.save_dir / "predictions.json"), "w") as f:
    #                 LOGGER.info(f"Saving {f.name}...")
    #                 json.dump(self.jdict, f)  # flatten and save
    #             stats = self.eval_json(stats)  # update stats
    #         if self.args.plots or self.args.save_json:
    #             LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
    #         return stats

    
class ZAxisMetrics(SimpleClass):
    """
    Utility class for computing detection metrics such as precision, recall, and mean average precision (mAP) of an
    object detection model.

    Args:
        save_dir (Path): A path to the directory where the output plots will be saved. Defaults to current directory.
        plot (bool): A flag that indicates whether to plot precision-recall curves for each class. Defaults to False.
        on_plot (func): An optional callback to pass plots path and data when they are rendered. Defaults to None.
        names (dict of str): A dict of strings that represents the names of the classes. Defaults to an empty tuple.

    Attributes:
        save_dir (Path): A path to the directory where the output plots will be saved.
        plot (bool): A flag that indicates whether to plot the precision-recall curves for each class.
        on_plot (func): An optional callback to pass plots path and data when they are rendered.
        names (dict of str): A dict of strings that represents the names of the classes.
        box (Metric): An instance of the Metric class for storing the results of the detection metrics.
        speed (dict): A dictionary for storing the execution time of different parts of the detection process.

    Methods:
        process(tp, conf, pred_cls, target_cls): Updates the metric results with the latest batch of predictions.
        keys: Returns a list of keys for accessing the computed detection metrics.
        mean_results: Returns a list of mean values for the computed detection metrics.
        class_result(i): Returns a list of values for the computed detection metrics for a specific class.
        maps: Returns a dictionary of mean average precision (mAP) values for different IoU thresholds.
        fitness: Computes the fitness score based on the computed detection metrics.
        ap_class_index: Returns a list of class indices sorted by their average precision (AP) values.
        results_dict: Returns a dictionary that maps detection metric keys to their computed values.
        curves: TODO
        curves_results: TODO
    """

    def __init__(self, save_dir=Path("."), plot=False, on_plot=None, names={}) -> None:
        """Initialize a DetMetrics instance with a save directory, plot flag, callback function, and class names."""
        self.save_dir = save_dir
        self.plot = plot
        self.on_plot = on_plot
        self.names = names
        self.box = Metric()
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.task = "detect"
        self.xyz_pairs = None
        

    def process(self, tp, conf, pred_cls, target_cls,xyz_pairs):
        """Process predicted results for object detection and update metrics."""
        results = ap_per_class(
            tp,
            conf,
            pred_cls, 
            target_cls,
            plot=self.plot,
            save_dir=self.save_dir,
            names=self.names,
            on_plot=self.on_plot,
        )[2:]
        self.all_box_results = {k:v for k,v in zip(["p", "r", "f1", "ap", "unique_classes", "p_curve", "r_curve", "f1_curve", "x", "prec_values"], results)}
        #p, r, f1, ap, unique_classes.astype(int), p_curve, r_curve, f1_curve,...
        box_results = [r[:,6] if i in (0,1,2,5,6,7) else r for i,r in enumerate(results)] # only single iou value for box results
        self.box.nc = len(self.names)
        self.box.update(box_results)
        self.num_iou_levels = tp.shape[1]
        fp = np.isnan(xyz_pairs[:,0,0])
        fn = np.isnan(xyz_pairs[:,1,0])
        self.fn = fn
        self.fp = fp
        self.xyz_pairs = xyz_pairs #filter out nans
        # self.z_pairs = z_pairs.transpose(1,2,0)
        # self.kpt_pairs = kpt_pairs.transpose(1,2,3,0)

    @property
    def keys(self):
        """Returns a list of keys for accessing specific metrics."""
        return ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)","metrics/Z-Axis rms.","metrics/xy-Axis rms."]

    def mean_results(self):
        """Calculate mean of detected objects & return precision, recall, mAP50, and mAP50-95."""
        results = self.box.mean_results()
        results.append(self.z_rms)
        results.append(self.xy_rms)

        return results

    # def class_result(self, i):
    #     """Return the result of evaluating the performance of an object detection model on a specific class."""
    #     return self.box.class_result(i)

    @property
    def maps(self):
        """Returns mean Average Precision (mAP) scores per class."""
        return self.box.maps

    @property
    def fitness(self):
        """Returns the fitness of box object."""
        return self.box.f1 * 1/self.z_rms * 1/self.xy_rms

    @property
    def ap_class_index(self):
        """Returns the average precision index per class."""
        return self.box.ap_class_index

    @property
    def results_dict(self):
        """Returns dictionary of computed performance metrics and statistics."""
        return dict(zip(self.keys + ["fitness"], self.mean_results() + [self.fitness]))

    @property
    def curves(self):
        """Returns a list of curves for accessing specific metrics curves."""
        return ["Precision-Recall(B)", "F1-Confidence(B)", "Precision-Confidence(B)", "Recall-Confidence(B)"]

    @property
    def curves_results(self):
        """Returns dictionary of computed performance metrics and statistics."""
        return self.box.curves_results
    
    
    @property
    def z_rms(self):
        """Return Z-Axis root mean square error [um] for 10 different IoU values"""
        z_pairs = self.xyz_pairs[:,:,2].T  # get z pairs
        z_distance = (np.subtract(*z_pairs))
        return np.sqrt(np.nanmean(z_distance**2))
    
    @property
    def xy_rms(self):
        """Return xy root mean square error [um] for 10 different IoU values"""
        distances = self.xy_distances
        rms = np.sqrt(np.nanmean(distances**2))
        return rms
    
    @property
    def xy_distances(self):

        x_pairs = self.xyz_pairs[:,:,0].T  # get x pairs
        y_pairs = self.xyz_pairs[:,:,1].T  # get y pairs
        x_distance = (np.subtract(*x_pairs))
        y_distance = (np.subtract(*y_pairs))
        distances = np.sqrt(x_distance**2+y_distance**2)

        return distances
    
def ap_per_class(
    tp, conf, pred_cls, target_cls, plot=False, on_plot=None, save_dir=Path(), names={}, eps=1e-16, prefix=""
):
    """
    Computes the average precision per class for object detection evaluation. Modified to compute for all iou values.

    Args:
        tp (np.ndarray): Binary array indicating whether the detection is correct (True) or not (False).
        conf (np.ndarray): Array of confidence scores of the detections.
        pred_cls (np.ndarray): Array of predicted classes of the detections.
        target_cls (np.ndarray): Array of true classes of the detections.
        plot (bool, optional): Whether to plot PR curves or not. Defaults to False.
        on_plot (func, optional): A callback to pass plots path and data when they are rendered. Defaults to None.
        save_dir (Path, optional): Directory to save the PR curves. Defaults to an empty path.
        names (dict, optional): Dict of class names to plot PR curves. Defaults to an empty tuple.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-16.
        prefix (str, optional): A prefix string for saving the plot files. Defaults to an empty string.

    Returns:
        tp (np.ndarray): True positive counts at threshold given by max F1 metric for each class.Shape: (nc,).
        fp (np.ndarray): False positive counts at threshold given by max F1 metric for each class. Shape: (nc,).
        p (np.ndarray): Precision values at threshold given by max F1 metric for each class. Shape: (nc,).
        r (np.ndarray): Recall values at threshold given by max F1 metric for each class. Shape: (nc,).
        f1 (np.ndarray): F1-score values at threshold given by max F1 metric for each class. Shape: (nc,).
        ap (np.ndarray): Average precision for each class at different IoU thresholds. Shape: (nc, 10).
        unique_classes (np.ndarray): An array of unique classes that have data. Shape: (nc,).
        p_curve (np.ndarray): Precision curves for each class. Shape: (nc, 1000).
        r_curve (np.ndarray): Recall curves for each class. Shape: (nc, 1000).
        f1_curve (np.ndarray): F1-score curves for each class. Shape: (nc, 1000).
        x (np.ndarray): X-axis values for the curves. Shape: (1000,).
        prec_values (np.ndarray): Precision values at mAP@0.5 for each class. Shape: (nc, 1000).
    """
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    x, prec_values = np.linspace(0, 1, 1000), []

    # Average precision, precision and recall curves
    ap, p_curve, r_curve = np.zeros((nc, tp.shape[1])), np.zeros((nc, tp.shape[-1], 1000)), np.zeros((nc, tp.shape[-1], 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r_curve[ci] = [np.interp(-x, -conf[i], recall[:, j], left=0) for j in range(tp.shape[-1])]  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p_curve[ci] = [np.interp(-x, -conf[i], precision[:, j], left=1) for j in range(tp.shape[-1])]  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if j == 0:
                prec_values.append(np.interp(x, mrec, mpre))  # precision at mAP@0.5

    prec_values = np.array(prec_values)  # (nc, 1000)

    # Compute F1 (harmonic mean of precision and recall)
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = dict(enumerate(names))  # to dict
    # if plot:
    #     plot_pr_curve(x, prec_values, ap, save_dir / f"{prefix}PR_curve.png", names, on_plot=on_plot)
    #     plot_mc_curve(x, f1_curve, save_dir / f"{prefix}F1_curve.png", names, ylabel="F1", on_plot=on_plot)
    #     plot_mc_curve(x, p_curve, save_dir / f"{prefix}P_curve.png", names, ylabel="Precision", on_plot=on_plot)
    #     plot_mc_curve(x, r_curve, save_dir / f"{prefix}R_curve.png", names, ylabel="Recall", on_plot=on_plot)

    i = np.array([smooth(f1_curve[:,j].mean(0), 0.1).argmax() for j in range(tp.shape[1])])  # max F1 index
    p, r, f1 = p_curve[:,np.arange(tp.shape[-1]),i], r_curve[:,np.arange(tp.shape[-1]), i], f1_curve[:,np.arange(tp.shape[-1]), i]  # max-F1 precision, recall, F1 values
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int), p_curve, r_curve, f1_curve, x, prec_values