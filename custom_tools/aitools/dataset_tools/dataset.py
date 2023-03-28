from pathlib import Path
import random
import warnings

class CustomDataset1:
    def __init__(self, data_dir, img_suffix=".png"):
        self.img_dir_list = [str(p.parent) for p in Path(data_dir).rglob("..")]
        self.img_suffix = img_suffix


    def split_dataset(self, train_p, val_p, test_p, cover=False, is_shuffle=False):
        assert train_p + val_p + test_p == 1.0
        for img_dir in self.img_dir_list:
            all_list = [str(img_path.stem) + "\n" for img_path in Path(img_dir).glob(f"*{self.img_suffix}")]
            if len(all_list) == 0:
                continue
            if is_shuffle:
                random.shuffle(all_list)
            val_num = int(val_p * len(all_list))
            test_num = int(test_p * len(all_list))
            train_num = len(all_list) - val_num - test_num
            s_i = 0
            for phase in ("train", "val", "test"):
                e_i = s_i + eval(f"{phase}_num")
                txt_path = str(Path(img_dir) / f"{phase}.txt")
                if Path(txt_path).exists():
                    if not cover:
                        warnings.warn(f"File exists: {txt_path}, skip it.")
                        break
                    else:
                        warnings.warn(f"File exists: {txt_path}, rewrite it.")
                with open(txt_path, "w") as f:
                    f.writelines(all_list[s_i: e_i])
                s_i = e_i
            print(f"{img_dir}: \ntrain: {train_num}, val: {val_num}, test:{test_num}, total:{len(all_list)}")

    def get_img_relative_paths(self, phase, img_suffix=".png"):
        img_relative_paths = []
        assert phase in ("train", "val", "test", "all")
        if phase == "all":
            for ph in ("train", "val", "test"):
                img_relative_paths.extend(self.get_img_relative_paths(ph, img_suffix))
            return img_relative_paths
        for img_dir in self.img_dir_list:
            txt_path = str(Path(img_dir) / f"{phase}.txt")
            if not Path(txt_path).exists():
                warnings.warn(f"{txt_path} is not exists, skip...")
                continue
            with open(txt_path) as f:
                img_relative_paths.extend(
                    [p.rstrip("\n") + img_suffix for p in f.readlines()]
                )
        return img_relative_paths
