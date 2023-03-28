from pycocotools.coco import COCO
from pathlib import Path

a = COCO(r"D:\personal_dataset\URPC_opticalimage_dataset\testA\annotation\testA.json")
b = COCO(r"D:\personal_dataset\URPC_opticalimage_dataset\testA\annotation\val.json")

c = 1

for i, (ann_a, ann_b) in enumerate(zip(a.anns.values(), b.anns.values())):
    if(ann_a["bbox"][:2] == ann_b["bbox"][:2] and abs(ann_a["bbox"][2] - ann_b["bbox"][2]) <= 1 and abs(ann_a["bbox"][3] - ann_b["bbox"][3]) <= 1):
        pass
    else:
        print(i)
        print(ann_a["image_id"], ann_b["image_id"])
        print(ann_a["category_id"], ann_a["bbox"], ann_a["area"])
        print(ann_b["category_id"], ann_b["bbox"], ann_b["area"])
        print("-" * 30)