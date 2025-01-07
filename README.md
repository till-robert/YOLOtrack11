# YOLOtrack11

extension of ultralytics' YOLOv11 at https://github.com/ultralytics/ultralytics for tracking of microscopy data

### Requirements

* ultralytics

```
pip install ultralytics
```

### How to use

See `zaxis_*.py/.ipynb`


### Notes

* This code mainly contains a series of patches to the original ultralytics framework to allow
    * passing around the extra parameters (z, center keypoint, angle,...) through the data pipelines
    * handling 16-bit monochrome images