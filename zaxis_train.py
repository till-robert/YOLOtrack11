from YOLOtrack11 import YOLOtrack11
from os import remove
# try:
#     remove("data_gen/Dataset/labels/train.cache")
#     remove("data_gen/Dataset/labels/val.cache")
# except:
#     pass
# print(model.model.model)
# print("loaded")
# for par in model.model.model.parameters():
#     par.requires_grad=False

# for par in model.model.model[-1].z_branch.parameters():
#     par.requires_grad = True
# def on_pretrain_routine_end (trainer):
#     for k, v in trainer.model.named_parameters():
#         # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
#         if("23" in k and "z_branch" not in k):
#             print(f"freezing layer '{k}'")
#             v.requires_grad = False

# model.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)

model = YOLOtrack11("yolo11n-zaxis.yaml")
results = model.train(data="datasets/ripples.yaml", imgsz=(640,540), background_level=2e4)#, z=5,freeze=22,box=0,cls=0,dfl=0,pose=0)
print("test")

    