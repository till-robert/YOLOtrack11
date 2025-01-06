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
results = model.train(data="../ultralytics/data_gen/dataset_hard.yaml", epochs=42, imgsz=512,)
print("test")