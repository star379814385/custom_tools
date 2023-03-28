import numpy as np

from .labelme_dataset import LabelmeData, LabelmeShape
from pathlib import Path
from xml.etree import ElementTree as ET
import imagesize
from tqdm import tqdm

def get_cate_bbox_list_by_xml_path(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annos = []
    for sub_node1 in root:
        if sub_node1.tag == "object":
            anno = dict()
            for sub_node2 in sub_node1:
                if sub_node2.tag == "name":
                    anno["category"] = sub_node2.text
                elif sub_node2.tag == "bndbox":
                    for n in sub_node2:
                        anno[n.tag] = float(n.text)
                else:
                    raise Exception()
            annos.append(anno)
        elif sub_node1.tag == "frame":
            pass
        else:
            raise Exception()
    return annos

# 仅用于目标检测
def urpc2labelme(label_dir, img_dir, img_pattern, save_dir=None):
    if save_dir is None:
        save_dir = img_dir
    img_paths = list(Path(img_dir).glob(img_pattern))
    for img_path in tqdm(img_paths):
        img_path = str(img_path)
        img_wh = imagesize.get(img_path)
        label_path = str(Path(img_path.replace(img_dir, label_dir)).with_suffix(".xml"))
        if not Path(label_path).exists():
            print(f"Label not found: {label_path}.")
        annos = get_cate_bbox_list_by_xml_path(label_path)

        labelme_data = LabelmeData(
            version="5.0.2",
            flags={},
            # shapes: Union[List[LabelmeShape], None] = None,
            shapes=[
                LabelmeShape(
                    label=ann["category"],
                    points=np.array(
                        [
                            [ann["xmin"], ann["ymin"]],
                            [ann["xmax"], ann["ymin"]],
                            [ann["xmax"], ann["ymax"]],
                            [ann["xmin"], ann["ymax"]],
                        ], dtype=np.float64
                    ).tolist(),  # 目标labelme2coco只支持polygon格式的标注
                    shape_type="polygon",
                    flags={}
                ) for ann in annos
            ],
            image_path=str(Path(img_path).relative_to(img_dir)),
            image_data=None,
            image_height=img_wh[1],
            image_width=img_wh[0],
            label_path=None,
        )
        labelme_data.save(str(Path(img_path.replace(img_dir, save_dir)).with_suffix(".json")))



