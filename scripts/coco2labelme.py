from custom_tools.aitools.labelme_tools import LabelmeDataset
from pycocotools.coco import COCO

coco_annotation_path = r"C:\Users\liruihui02\Downloads\Compressed\annotations\instances_default.json"
save_dir = r"D:\project\6_qiangban_qiti\dataset\images\墙板（未标注）"
labelme_dataset = LabelmeDataset.build_from_coco_annotations(
    coco_annotations=COCO(coco_annotation_path)
)
labelme_dataset.save(save_dir=save_dir)