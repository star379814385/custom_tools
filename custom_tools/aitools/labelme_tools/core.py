import copy
from ...pyutils.fileio import load_json, dump_json
from typing import Union, List
from pathlib import Path


class LabelmeShape:
    def __init__(self, label=None, points=None, group_id=None, shape_type=None, flags=None):
        self.label = label
        self.points = points
        self.group_id = group_id
        self.shape_type = shape_type
        self.flags = flags


class LabelmeData:
    def __init__(
        self,
        version: Union[str, None] = None,
        flags: Union[dict, None] = None,
        shapes: Union[List[LabelmeShape], None] = None,
        image_path: Union[str, None] = None,
        image_data: Union[str, None] = None,
        image_height: Union[int, None] = None,
        image_width: Union[int, None] = None,
        label_path: Union[str, None] = None
    ) -> None:
        self.version = version
        self.flags = flags
        self.shapes = shapes
        self.image_path = image_path
        self.image_data = image_data
        self.image_height = image_height
        self.image_width = image_width
        self.label_path = label_path

    def mapping_categories(self, mapping: dict):
        for i in range(len(self.shapes)):
            new_label = mapping.get(self.shapes[i].label, None)
            if new_label is not None:
                self.shapes[i].label = new_label

    def delete_by_categories(self, categories):
        self.shapes = [ann for ann in self.shapes if ann.label not in categories]

    def remain_by_categories(self, categories):
        self.shapes = [ann for ann in self.shapes if ann.label in categories]



    def save(self, save_path):
        json_datas = {
            "version": self.version,
            "flags": self.flags,
            "shapes": [
                {
                    "label": ann.label,
                    "points": ann.points,
                    "group_id": ann.group_id,
                    "shape_type": ann.shape_type,
                    "flags": ann.flags,
                } for ann in self.shapes
            ],
            "imagePath": self.image_path,
            "imageData": self.image_data,
            "imageHeight": self.image_height,
            "imageWidth": self.image_width,
        }
        dump_json(json_datas, save_path)

    @classmethod
    def build_by_json_path(cls, json_path, remain_image_data=False, remain_label_path=True):
        json_datas: dict = load_json(json_path)
        json_datas["shapes"] = [LabelmeShape(**ann) for ann in json_datas["shapes"]]
        json_datas["image_path"] = json_datas.pop("imagePath")
        if remain_image_data:
            json_datas["image_data"] = json_datas.pop("imageData")
        else:
            json_datas.pop("imageData")
        json_datas["image_height"] = json_datas.pop("imageHeight")
        json_datas["image_width"] = json_datas.pop("imageWidth")
        if remain_label_path:
            json_datas["label_path"] = json_path
        return cls(**json_datas)

class LabelmeDataset:
    def __init__(self, data_list: List[LabelmeData]):
        self.data_list = data_list

    def get_categories_hist(self):
        categories_hist = dict()
        for data in self.data_list:
            for ann in data.shapes:
                if categories_hist.get(ann.label, None) is None:
                    categories_hist[ann.label] = 1
                else:
                    categories_hist[ann.label] += 1
        return categories_hist

    @classmethod
    def build_from_json_paths(cls, json_paths, **kwargs):
        data_list = []
        for json_path in json_paths:
            json_path = str(json_path)
            try:
                data = LabelmeData.build_by_json_path(json_path, **kwargs)
            except Exception as e:
                print(e)
                continue
            data_list.append(data)
        return cls(data_list)

    @classmethod
    def build_from_json_dir(cls, json_dir, is_recursive=True, **kwargs):
        json_paths = Path(json_dir).rglob("*.json") if is_recursive else Path(json_dir).glob("*.json")
        json_paths = list(json_paths)
        return cls.build_from_json_paths(json_paths, **kwargs)