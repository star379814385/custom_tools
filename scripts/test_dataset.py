from custom_tools.aitools.dataset_tools.dataset import CustomDataset1

if __name__ == "__main__":
    img_suffix = ".jpg"
    img_dir = r"D:\personal_dataset\URPC_opticalimage_dataset\train\image"
    dataset = CustomDataset1(img_dir, img_suffix=img_suffix)
    dataset.split_dataset(train_p=0.8, val_p=0.2, test_p=0, cover=True)
    img_paths = dataset.get_img_relative_paths("train", img_suffix=img_suffix)
    print(img_paths)