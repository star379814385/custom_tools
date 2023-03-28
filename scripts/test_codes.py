from custom_tools.aitools.labelme_tools import LabelmeDataset
from custom_tools.pyutils import dump_json

if __name__ == "__main__":
    json_dir = r"D:\project\6_qiangban_qiti\dataset\images\砌体"
    # json_dir = r"D:\project\6_qiangban_qiti\dataset\images\金地博裕花园-2栋-6层-初测-砌筑工程-自检"
    labelme_dataset = LabelmeDataset.build_from_json_dir(json_dir, is_recursive=True)
    # # 0.把imageData词条去除
    # for data in labelme_dataset.data_list:
    #     data.save(data.label_path)
    # 1.常规检查数据
    # 数据是否有问题？
    labelme_dataset.check_data_list(img_suffix=".png", filter_data=False)
    # 2.确认数据相关信息
    # 按照类别-标注类型对查看标注类别
    print(labelme_dataset.get_categories_shape_types_hist())
    # 2.1更改不正确的标注
    mapping_dict = {
        "qiti_out": "qiti",
        "qiti_window": "qiti",
    }
    labelme_dataset.mapping_categories(mapping_dict)
    print(labelme_dataset.get_categories_shape_types_hist())
    # 2.2 删除不需要的标注
    # 3.生成其他格式标注（coco）
    categories = ("qiti", )
    coco_annotations = labelme_dataset.get_coco_annotations(
        categories=categories,
        remove_empty=True,
        strict=False
    )
    # dump_json(coco_annotations, "tmp.json")
