from YOLOtrack11 import YOLOtrack11
# from os import remove
# try:
#     remove("data_gen/Dataset/labels/train.cache")
#     remove("data_gen/Dataset/labels/val.cache")
# except:
#     pass
model = YOLOtrack11("../ultralytics/runs/pose/train23/weights/last.pt")
# print(model.model.model)
print("loaded")
# results = model.train(data="data_gen/data.yaml", epochs=100, imgsz=512,)
results = model.val()

print("test")