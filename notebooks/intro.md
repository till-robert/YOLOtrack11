# Welcome to the YOLO bacteria tracking documentation!

## What is YOLO?

* Deep learning object detector and classifier
* Single-Shot model, based on anchor points
* Supervised learning, i.e. regression to training data

## Why use YOLO?


* It's fast (< 30ms per image)
* flexible (easily add extra parameters like orientation, shape detection, intensity)
* GPU-friendly


## Getting started

YOLOtrack11 is an extension of [YOLO11](https://docs.ultralytics.com) which uses PyTorch.

This scheme uses a [training data generator](./ImageGenerator), together with a reference stack, to create artificial images and corresponding label files. This way, there is no manual labelling necessary. For this approach to work, the artificial dataset images must very closely resemble the real microscopy images, including imperfections. Machine learning models perform generally quite poorly on data lying outside its training domain.

After generating training images, the model needs to be trained:

```python
from YOLOtrack11 import YOLOtrack11

model = YOLOtrack11("yolo11n-zaxis.yaml") # load the zaxis model with scale 'n'
results = model.train(data="datasets/dataset.yaml", epochs=40, <other-arguments>) # train the model for 40 epochs
```

The training progress is saved in the `runs/` directory. The functions in `plot_results.ipynb` provide a convenient way to review the training progress.

The other arguments may be any of the parameters that are defined in the file `default.yaml`.

Please click on a link to the side to continue.
