import copy
import logging
import shutil

import cv2

from ...pyutils.fileio import load_json, dump_json, create_directory_if_parent_not_exist
from typing import Union, List
from pathlib import Path
import numpy as np
import imagesize
import warnings
from tqdm import tqdm
from pycocotools.coco import COCO
from scipy.io import loadmat


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
        self.points_npy = np.array(self.points, dtype=np.float64).reshape((-1, 2))

    def get_hw(self):
        if self.points_npy is None:
            self.update_points_npy()
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
        self.version = version if version is not None else "5.0.2"
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



    def check_data(self, img_suffix=".png"):
        # label_path 和 image_path是否对应
        if Path(self.label_path).stem != Path(self.label_path).stem:
            error = "图像名称与标注名称不一致"
            return False, error
        image_path = str(Path(self.label_path).with_suffix(img_suffix))
        if not Path(image_path).exists():
            error = "不存在对应图像数据"
            return False, error
        w, h = imagesize.get(image_path)
        if w != self.image_width or h != self.image_height:
            error = "对应图像尺寸与标注数据不一致"
            return False, error
        return True, None




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
        # todo: create dir if need
        create_directory_if_parent_not_exist(save_path)
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

    def get_categories_shape_types_hist(self):
        categories_shape_types = dict()
        for data in self.data_list:
            for ann in data.shapes:
                cate_shape_type = (ann.label, ann.shape_type)
                if categories_shape_types.get(cate_shape_type, None) is None:
                    categories_shape_types[cate_shape_type] = 1
                else:
                    categories_shape_types[cate_shape_type] += 1
        return categories_shape_types

    def find_json_paths(self, category:Union[tuple, None] = None, shape_type: Union[tuple, None] = None):
        # 获取同时符合以上条件的标注文件，一般为了二次确认标注
        for data in self.data_list:
            ids = list(range(len(data.shapes)))
            if category is not None:
                ids = [id_ for id_ in ids if data.shapes[id_].label == category]
            if shape_type is not None:
                ids = [id_ for id_ in ids if data.shapes[id_].shape_type == shape_type]
            if len(ids) > 0:
                print(data.label_path)

    def check_data_list(self, img_suffix, filter_data=False):
        flag_list = []
        for data in self.data_list:
            flag, error = data.check_data(img_suffix)
            if not flag:
                print(f"{error}: {data.label_path}")
            flag_list.append(flag)
        if filter_data:
            self.data_list = [data for data, flag in zip(self.data_list, flag_list) if flag]

    def mapping_categories(self, mapping: dict):
        for i in range(len(self.data_list)):
            self.data_list[i].mapping_categories(mapping)

    def get_categories_ids(self, categories=None):
        if categories is None:
            categories = set()
            for data in self.data_list:
                for ann in data.shapes:
                    categories.add(ann.label)
            categories = tuple(categories)
        categories_ids = {cate: i for i, cate in enumerate(categories)}
        return categories_ids

    def get_coco_annotations(self, categories=None, remove_empty=True, strict=True):
        categories_ids = self.get_categories_ids(categories)
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
        for data_id, data in enumerate(self.data_list):
            cur_annotations = []
            cate_contours_list = data.get_categories_contours()
            for cate_contours in cate_contours_list:
                cate_id = categories_ids.get(cate_contours["label"], None)
                if cate_id is None:
                    # 标注存在多余的类别标注
                    if strict:
                        raise Exception("存在多余类别标注: {}".format(cate_contours["label"]))
                    else:
                        warnings.warn("存在多余类别标注: {}".format(cate_contours["label"]))
                        continue
                contours = cate_contours["contours"]
                ps = np.concatenate(contours, axis=0)
                x, y, w, h = cv2.boundingRect(ps.astype(np.float32))
                w -= 1
                h -= 1
                if w < 1 or h < 1:
                    continue
                area = sum([cv2.contourArea(contour.astype(np.float32)) for contour in contours])
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
            if len(cur_annotations) == 0 and remove_empty:
                warnings.warn(f"不存在有效标注: data_id = {data_id}")
                continue
            cur_image = {
                "id": image_id,
                "width": data.image_width,
                "height": data.image_height,
                "file_name": data.image_path
            }
            coco_annotations["images"].append(cur_image)
            coco_annotations["annotations"].extend(cur_annotations)
            image_id += 1
        return coco_annotations


    def save_coco_annotations(self, save_path, **kwargs):
        coco_annotations = self.get_coco_annotations(**kwargs)
        create_directory_if_parent_not_exist(save_path)
        dump_json(coco_annotations, save_path)

    def get_yolo_datas(self, categories=None, remove_empty=True, strict=True):
        categories_ids = self.get_categories_ids(categories)
        print(f"categories_ids is {categories_ids}.")
        """
        img1: {
            image_path, 
            label_data,
        }
        """
        yolo_datas = dict()
        for data_id, data in enumerate(self.data_list):
            stem = str(Path(data.image_path).stem)
            if yolo_datas.get(stem, None) is not None:
                print("存在相同名称的图像数据:\n{}\n{}".format(yolo_datas[stem]["image_path"], data.image_path))
                if strict:
                    print(f"Remove {data.image_path}.")
                else:
                    raise Exception()
            else:
                label_data_list = []
                cate_contours_list = data.get_categories_contours()
                for cate_contours in cate_contours_list:
                    cate_id = categories_ids.get(cate_contours["label"], None)
                    if cate_id is None:
                        # 标注存在多余的类别标注
                        if strict:
                            raise Exception("存在多余类别标注: {}".format(cate_contours["label"]))
                        else:
                            warnings.warn("存在多余类别标注: {}".format(cate_contours["label"]))
                            continue
                    contours = cate_contours["contours"]
                    ps = np.concatenate(contours, axis=0)
                    x, y, w, h = cv2.boundingRect(ps.astype(np.float32))
                    w -= 1
                    h -= 1
                    if w < 1 or h < 1:
                        continue
                    cxp = (x + w / 2) / data.image_width
                    cyp = (y + h / 2) / data.image_height
                    wp = w / data.image_width
                    hp = h / data.image_height
                    label_data_list.append((cate_id, cxp, cyp, wp, hp, ))
                if len(label_data_list) == 0 and remove_empty:
                    warnings.warn(f"不存在有效标注: data_id = {data_id}")
                    continue


                yolo_datas[stem] = {
                    "image_path": data.image_path,
                    "label_data": label_data_list
                }

        return yolo_datas

    # def save_yolo_dataset(self, save_dir, phase, **kwargs):
    #     yolo_datas = self.get_yolo_datas(**kwargs)
    #     name_list = []
    #     for stem, yolo_data in tqdm(yolo_datas.items()):
    #         # src_image_path = str(Path(data_root) / yolo_data["image_path"])
    #         label_txt_data = []
    #         for cate_id, x, y, w, h in yolo_data["label_data"]:
    #             label_txt_data.append(f"{cate_id} {x} {y} {w} {h}\n")
    #         # dst_image_path = str(Path(save_dir) / "images" / Path(src_image_path).name)
    #         dst_label_path = str(Path(save_dir) / f"{stem}.txt")
    #         # create_directory_if_parent_not_exist(dst_image_path)
    #         create_directory_if_parent_not_exist(dst_label_path)
    #         # if not Path(dst_image_path).exists():
    #         #     shutil.copy(src_image_path, dst_image_path)
    #         with open(dst_label_path, "w") as f:
    #             f.writelines(label_txt_data)
    #         name_list.append(Path(yolo_data["image_path"]).name + "\n")
    #     save_txt_path = Path(save_dir) / f"{phase}.txt"
    #     with open(save_txt_path, "w") as f:
    #         f.writelines(name_list)

    def save_yolo_dataset(self, data_root, save_dir, **kwargs):
        yolo_datas = self.get_yolo_datas(**kwargs)
        name_list = []
        for stem, yolo_data in tqdm(yolo_datas.items()):
            src_image_path = str(Path(data_root) / yolo_data["image_path"])
            label_txt_data = []
            for cate_id, x, y, w, h in yolo_data["label_data"]:
                label_txt_data.append(f"{cate_id} {x} {y} {w} {h}\n")
            dst_image_path = str(Path(save_dir) / "images" / Path(src_image_path).name)
            dst_label_path = str(Path(save_dir) / "labels" / f"{stem}.txt")
            create_directory_if_parent_not_exist(dst_image_path)
            create_directory_if_parent_not_exist(dst_label_path)
            if not Path(dst_image_path).exists():
                shutil.copy(src_image_path, dst_image_path)
            with open(dst_label_path, "w") as f:
                f.writelines(label_txt_data)
            name_list.append(Path(yolo_data["image_path"]).name + "\n")

    def save(self, save_dir, ):
        print(len(self.data_list))
        # 使用label_path为相对路径
        for data in self.data_list:
            assert data.label_path is not None
        for data in tqdm(self.data_list):
            save_path = str(Path(save_dir) / data.label_path)
            data.save(save_path)


    @classmethod
    def build_from_json_paths(cls, json_paths, **kwargs):
        print("loading datas......")
        data_list = []
        for json_path in tqdm(json_paths):
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

    @classmethod
    def build_from_coco_annotations(cls, coco_annotations: COCO):
        data_list = []
        categories = {}
        for cat in coco_annotations.cats.values():
            categories[cat["id"]] = cat["name"]
        assert coco_annotations.imgToAnns.keys() == coco_annotations.imgs.keys()
        for i in coco_annotations.imgs.keys():
            img_info = coco_annotations.imgs[i]
            anns_info = coco_annotations.imgToAnns[i]
            if len(anns_info) > 0:
                assert img_info["id"] == anns_info[0]["image_id"]
            labelme_data = LabelmeData(
                version=None,
                flags={},
                shapes=[
                    LabelmeShape(
                        label=categories[ann_info["category_id"]],
                        points=[[ps[pi * 2], ps[pi * 2 + 1]] for pi in range(len(ps) // 2)],
                        group_id=gi,
                        shape_type="polygon",
                        flags={},
                    ) for gi, ann_info in enumerate(anns_info) for ps in ann_info["segmentation"]
                ],
                image_path=Path(img_info["file_name"]).name,
                image_data=None,
                image_height=img_info["height"],
                image_width=img_info["width"],
                label_path=str(Path(img_info["file_name"]).with_suffix(".json"))
            )
            data_list.append(labelme_data)
        return cls(data_list)

    @classmethod
    def build_from_BITVehicle_Dataset(cls, data_dir):
        mat_path = str(Path(data_dir) / "VehicleInfo.mat")
        import h5py
        
        f = h5py.File(mat_path) 
        mat_data = loadmat(mat_path)
        labelme_data_list = []
        cls_set = set()
        for data in mat_data["VehicleInfo"]:
            assert len(data) == 1
            for img_names, hs, ws, all_bboxes, all_cate_ids in data:
                annos = []
                img_name = None
                h = None
                w = None
                for img_name_, h_, w_, bboxes, cate_ids in zip(img_names, hs, ws, all_bboxes, all_cate_ids):
                    assert isinstance(img_name_, str)
                    img_name = img_name_
                    # print(img_name)
                    # print(h)
                    assert len(h_) == 1 and len(w_) == 1
                    h = int(h_[0])
                    w = int(w_[0])
                    for (x1, y1, x2, y2, cate_name), cate_id in zip(bboxes, cate_ids):
                        # print(cate_name)
                        assert len(cate_name) == 1
                        assert len(x1) == 1 
                        assert len(y1) == 1 
                        assert len(x2) == 1 
                        assert len(y2) == 1 
                        cate_name = cate_name[0]
                        # cls_dict[cate_id] = cate_name
                        cls_set.add((cate_id, cate_name))
                        x1 = int(x1[0][0])
                        y1 = int(y1[0][0])
                        x2 = int(x2[0][0])
                        y2 = int(y2[0][0])
                        annos.append(
                            LabelmeShape(
                                label=cate_name, 
                                points=[[x1, y1], [x1, y2], [x2, y2], [x2, y1]],
                                shape_type="polygon", 
                                flags={}, 
                                )
                            )
                assert len(annos) == 1
                labelme_data = LabelmeData(
                    version=None, 
                    flags={}, 
                    shapes=annos, 
                    image_path=img_name, 
                    image_data=None, 
                    image_height=h, 
                    image_width=w, 
                    label_path=str(Path(img_name).with_suffix(".json"))
                )
                labelme_data_list.append(labelme_data)
        # print(cls_dict)
        print(cls_set)
        return cls(labelme_data_list)