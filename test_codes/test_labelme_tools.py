from custom_tools.aitools.labelme_tools import LabelmeData, LabelmeDataset

if __name__ == "__main__":
    data_dir = "/home/rainfylee/desktop/dataset/qiangban/images"
    labelme_dataset = LabelmeDataset.build_from_json_dir(data_dir, is_recursive=True, remain_image_data=True)
    print(labelme_dataset.get_categories_hist())