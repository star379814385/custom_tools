from custom_tools.aitools.dataset_tools.dataset import CustomDataset1

# data_dir = r"D:\project\6_qiangban_qiti\dataset\images\墙板"
data_dir = r"D:\project\6_qiangban_qiti\dataset\images\墙板砌体"
img_suffix = r".png"
dataset = CustomDataset1(data_dir, img_suffix)
dataset.split_dataset(train_p=0.8, val_p=0.2, test_p=0, is_shuffle=True, cover=True)