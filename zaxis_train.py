from YOLOtrack11 import YOLOtrack11
from os import remove
# try:
#     remove("data_gen/Dataset/labels/train.cache")
#     remove("data_gen/Dataset/labels/val.cache")
# except:
#     pass
model = YOLOtrack11("yolo11n-zaxis.yaml")
# print(model.model.model)
print("loaded")
results = model.train(data="datasets/dataset_yundon.yaml", epochs=18, imgsz=640)
print("test")