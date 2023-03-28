from custom_tools.aitools.labelme_tools import LabelmeDataset
from pathlib import Path
from custom_tools.aitools.dataset_tools.dataset import CustomDataset1

class Qiangban:
    data_root = r"D:\project\6_qiangban_qiti\dataset\images"
    data_dirs = [
        r"D:\project\6_qiangban_qiti\dataset\images\墙板\63亩墙板检测",
        r"D:\project\6_qiangban_qiti\dataset\images\墙板\0608_rencaifang",
        r"D:\project\6_qiangban_qiti\dataset\images\墙板砌体\63亩墙板砌体",
    ]


    hard_dirs = [
        # 白色墙板
        r"D:\project\6_qiangban_qiti\dataset\images\墙板\南阳邓州花洲府项目-6栋-12层-初测",
        r"D:\project\6_qiangban_qiti\dataset\images\墙板\A0007-000442 中铁城建西客站片区腊山河西侧3地块项目",
        r"D:\project\6_qiangban_qiti\dataset\images\墙板砌体\A0009-000179 洪泽江山府",
        r"D:\project\6_qiangban_qiti\dataset\images\墙板砌体\A0009-000127 水沐云顶花园三标",
        r"D:\project\6_qiangban_qiti\dataset\images\墙板砌体\A0002-000590 盐城市城北星钻花苑",
        r"D:\project\6_qiangban_qiti\dataset\images\墙板砌体\A0002-000430 巴城项目"
    ]
    hard_sample_repeats = [2, 2, 2, 2, 2, 2]
    # hard_sample_repeats = [2, 2, 2, 3]
    categories = ("qiangban", )
    annotations_dir = r"D:\project\6_qiangban_qiti\dataset\annotations\qiangban_20230322"
    # 部分错误标注类别可以借助该参数更改类别
    category_mapping = {
        "qiangban_out": "qiti",
    }
    use_empty = False

if __name__ == "__main__":
    cfg = Qiangban
    img_suffix = ".png"
    data_dirs = cfg.data_dirs
    categories = cfg.categories
    dataset_list = CustomDataset1()
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
