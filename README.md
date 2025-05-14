# YOLOtrack11

extension of ultralytics' YOLOv11 at https://github.com/ultralytics/ultralytics for tracking of microscopy data

### Requirements

* ultralytics

```
pip install ultralytics
```

### How to use

See `notebooks/zaxis_*.py/.ipynb` and `notebooks/ImageGenerator.ipynb`, and the [Documentation](https://till-robert.github.io/YOLOtrack11/intro.html)

### Notes

* This code contains a series of patches to the original ultralytics framework to allow
    * passing around the extra parameters (z, center keypoint, angle,...) through the data pipelines
    * handling 16-bit monochrome images

* Secondly, a new task (`zaxis`) is defined (see `__init__.py` and `model.py`). It includes
    * a new model called `ZAxisModel` with the modified head module (`ZAxis`) that inherits from the `Pose` estimation head. It includes an additional branch, `z_branch`, which handles the detection of the additional parameters (e.g. `z`).
    * a new loss function for the `z_branch`, specified in `loss.py`.
    * modified `Trainer`, `Predictor` and `Validator` classes for the `zaxis` task. They include mostly small adjustments, except the new `Validator`-class (located in `val.py`), which contains the new validation metrics, like the calculations for the $RMS_{xy}$ and $RMS_z$.


