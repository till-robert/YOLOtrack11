from YOLOtrack11 import YOLOtrack11
model = YOLOtrack11("../ultralytics/runs/pose/train122/weights/best.pt")

model.export(format="engine")