from custom_tools.aitools.labelme_tools import LabelmeDataset
from custom_tools.pyutils import dump_json
from custom_tools.aitools.dataset_tools.dataset import CustomDataset1
from pathlib import Path

if __name__ == "__main__":
    img_suffix = ".jpg"
    data_dir = r"D:\personal_dataset\URPC_opticalimage_dataset\testA\image"
    save_annotation_dir = r"D:\personal_dataset\URPC_opticalimage_dataset\testA\annotation"
    categories = ("holothurian", "echinus", "scallop", "starfish")
    dataset = CustomDataset1(data_dir, img_suffix=img_suffix)
    # dataset.split_dataset(train_p=0, val_p=1, test_p=0)
    for phase in ("train", "val", "test", "all"):
    # for phase in ("all", ):
        paths = dataset.get_img_relative_paths(phase, img_suffix=img_suffix)
        labelme_dataset = LabelmeDataset.build_from_json_paths(
            json_paths=[str((Path(data_dir) / p).with_suffix(".json")) for p in paths]
        )
        # # 0.把imageData词条去除
        # for data in labelme_dataset.data_list:
        #     data.save(data.label_path)
        # 1.常规检查数据
        # 数据是否有问题？
        labelme_dataset.check_data_list(img_suffix=".jpg", filter_data=False)
        # 2.确认数据相关信息
        # 按照类别-标注类型对查看标注类别
        print(labelme_dataset.get_categories_shape_types_hist())
        # 2.1更改不正确的标注
        # mapping_dict = {
        #     "qiti_out": "qiti",
        #     "qiti_window": "qiti",
        # }
        # labelme_dataset.mapping_categories(mapping_dict)
        # print(labelme_dataset.get_categories_shape_types_hist())
        # 2.2 删除不需要的标注
        # 3.生成其他格式标注（coco）
        coco_annotations = labelme_dataset.get_coco_annotations(
            categories=categories,
            remove_empty=False,
            strict=False
        )
        save_annotation_path = str(Path(save_annotation_dir) / f"{phase}.json")
        if not Path(save_annotation_path).parent.exists():
            Path(save_annotation_path).parent.mkdir(parents=True)
        dump_json(coco_annotations, save_annotation_path)