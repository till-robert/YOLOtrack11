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
results = model.train(data="../ultralytics/data_gen/dataset.yaml", epochs=18, imgsz=512,)
print("test")