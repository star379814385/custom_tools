import numpy as np

from .evaluator import InsSegEvaluator
from ..cvutils import read_image
from pathlib import Path
from .labelme_tools.labelme_dataset import LabelmeData

class Tester:
    def __init__(self):
        pass

    def run(self, paths):
        pass

    def _run(self, path):
        pass

    def _after_run(self):
        pass

    def get_label_by_path(self, path):
        pass

    def get_image_by_path(self, path):
        pass

    def data_pipeline(self, results: dict):
        return results


class InsSegTester(Tester):
    def __init__(self, categories):
        super(InsSegTester, self).__init__()
        self.evaluator = InsSegEvaluator()
        self.categories = {cate: i for i, cate in enumerate(categories)}

    def _run(self, path: str):
        img = self.get_image_by_path(path)
        if img is None:
            return None
        label = self.get_label_by_path(path)


    def get_image_by_path(self, path):
        try:
            img = read_image(path)
        except Exception as e:
            print(e)
            return None
        return img

    def get_label_by_path(self, path):
        # 允许标注不存在，此时标注为空
        # 该标注类型默认为labelme
        label_path = Path(path).with_suffix(".json")
        label_data = LabelmeData.build_by_json_path(label_path)
        cate_contours_list = label_data.get_categories_contours()
        masks = np.zeros(())


