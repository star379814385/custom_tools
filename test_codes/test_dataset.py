from custom_tools.aitools.dataset_tools.dataset import CustomDataset1

if __name__ == "__main__":
    img_dir = r"D:\project\6_qiangban_qiti\dataset\images\砌体\A0022-000001 金地博裕花园\金地博裕花园-5栋-3层-初测-砌筑工程-自检"
    dataset = CustomDataset1(img_dir)
    dataset.split_dataset(train_p=0.8, val_p=0.2, test_p=0, cover=True)
    img_paths = dataset.get_img_relative_paths("train")
    print(img_paths)