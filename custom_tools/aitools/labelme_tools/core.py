import copy

import cv2

from ...pyutils.fileio import load_json, dump_json
from typing import Union, List
from pathlib import Path
import numpy as np


class LabelmeShape:
    def __init__(self, label=None, points=None, group_id=None, shape_type=None, flags=None):
        self.label = label
        self.points = points
        self.group_id = group_id
        self.shape_type = shape_type
        self.flags = flags
        self.points_npy = None
        self.hw = None
        self.area = None

    def update_points_npy(self):
        return np.array(self.points, dtype=np.float64).reshape((-1, 2))

    def get_hw(self):
        if self.points_npy is None:
            self.points_npy = self.update_points_npy()
        h, w = np.ptp(self.points_npy, axis=0)
        return h, w


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

    def delete_by_shape_types(self, shape_types):
        self.shapes = [ann for ann in self.shapes if ann.shape_type not in shape_types]

    def remain_by_shape_types(self, shape_types):
        self.shapes = [ann for ann in self.shapes if ann.shape_type in shape_types]

    def delete_by_height_width(self, height_range=(0, np.inf), width_range=(0, np.inf)):
        new_shapes = []
        for ann in self.shapes:
            h, w = ann.get_hw()
            if h < height_range[0] or h > height_range[1]:
                continue
            if w < width_range[0] or w > width_range[1]:
                continue
            new_shapes.append(ann)
        self.shapes = new_shapes

    def merge_group_ids(self):
        for i in range(len(self.shapes)):
            self.shapes[i].group_id = None

    def get_categories_contours(self):
        cate_contours_list = []
        cate_group_id_to_contours = dict()
        for ann in self.shapes:
            contour = np.array(ann.points, dtype=np.float64).reshape((-1, 2))
            if ann.group_id is None:
                cate_contours_list.append(
                    {
                        "label": ann.label,
                        "contours": [contour],
                    }
                )
            else:
                cate_group_id = (ann.label, ann.group_id)
                if cate_group_id not in cate_group_id_to_contours.keys():
                    cate_group_id_to_contours[cate_group_id] = [contour]
                else:
                    cate_group_id_to_contours[cate_group_id].append(contour)
        for (cate, group_id), contours in cate_group_id_to_contours.items():
            cate_contours_list.append(
                {
                    "label": cate,
                    "contours": contours
                }
            )
        return cate_contours_list

    def check_data(self, data_root):
        pass




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

    def check_data_list(self):
        pass

    def get_coco_annotations(self, categories=None):
        if categories is None:
            categories = set()
            for data in self.data_list:
                for ann in data.shapes:
                    categories.add(ann.label)
            categories = tuple(categories)
        categories_ids = {cate: i for i, cate in enumerate(categories)}
        print(f"categories_ids is {categories_ids}.")

        coco_annotations = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        # get categories
        for cate, i in categories_ids.items():
            coco_annotations["categories"].append(
                {
                    "id": i,
                    "name": cate
                }
            )
        # get images and annotations
        image_id = 0
        annotation_id = 0
        for data in self.data_list:
            cur_image = {
                "id": image_id,
                "width": data.image_width,
                "height": data.image_height,
                "file_name": data.image_path
            }
            cur_annotations = []
            cate_contours_list = data.get_categories_contours()
            for cate_contours in cate_contours_list:
                cate_id = categories_ids.get(cate_contours["label"], None)
                if cate_id is None:
                    continue
                contours = cate_contours["contours"]
                ps = np.concatenate(contours, axis=0)
                x, y, w, h = cv2.boundingRect(ps)
                area = sum([cv2.contourArea(contour) for contour in contours])
                ann = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": cate_id,
                    "segmentation": [list(contour.flatten()) for contour in contours],
                    "area": area,
                    "bbox": [x, y, w, h],
                    "iscrowd": 0
                }
                cur_annotations.append(ann)
                annotation_id += 1
            coco_annotations["images"].append(cur_image)
            coco_annotations["annotations"].extend(cur_annotations)
            image_id += 1
        return coco_annotations

    def save_as_coco_annotations(self, categories, save_path):
        coco_annotations = self.get_coco_annotations(categories)
        if not Path(save_path).parent.exists():
            Path(save_path).parent.mkdir(parents=True)
        dump_json(coco_annotations, save_path)


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

