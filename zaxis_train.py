from YOLOtrack11 import YOLOtrack11
from os import remove
# try:
#     remove("data_gen/Dataset/labels/train.cache")
#     remove("data_gen/Dataset/labels/val.cache")
# except:
#     pass
model = YOLOtrack11("../ultralytics/runs/pose/train106/weights/best.pt")
# model = YOLOtrack11("yolo11n-zaxis.yaml")
# print(model.model.model)
# print("loaded")
# for par in model.model.model.parameters():
#     par.requires_grad=False

# for par in model.model.model[-1].z_branch.parameters():
#     par.requires_grad = True

results = model.train(data="datasets/dataset_yundon_fine_tune.yaml", epochs=20, imgsz=(640,540),warmup_epochs = 0, z=10000)#,box=0,cls=0,dfl=0,pose=0)
print("test")

