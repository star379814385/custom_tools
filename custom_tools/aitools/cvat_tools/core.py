from pathlib import Path
import shutil
from custom_tools.pyutils import dump_json

class Cvat:
    def __init__(self, src_dir, dst_dir):
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.src_paths = [str(p) for p in Path(src_dir).rglob("*.png")]
        self.dst_paths = self.get_dst_paths()

    def copy_src_to_dst(self):
        json_datas = dict(
            src_dir=self.src_dir,
            dsr_dir=self.dst_dir,
            mapping=[],
        )
        for src, dst in zip(self.src_paths, self.dst_paths):
            if not Path(dst).parent.exists():
                Path(dst).parent.mkdir(parents=True)
            shutil.copy(src, dst)
            json_datas["mapping"].append(dict(src=src, dst=dst))
        save_json_path = str(Path(self.dst_dir) / "src2dst.json")
        if not Path(save_json_path).parent.exists():
            Path(save_json_path).parent.mkdir(parents=True)
        dump_json(json_datas, save_json_path)

    def copy_dst_labelme_to_src(self):
        for src, dst in zip(self.src_paths, self.dst_paths):
            dst = str(Path(dst).with_suffix(".json"))
            src = str(Path(src).with_suffix(".json"))
            shutil.copy(dst, src)

    def get_dst_paths(self):
        dst_paths = []
        for i, src_path in enumerate(self.src_paths):
            # dst_paths.append(str(Path(self.dst_dir) / f"{i}{Path(src_path).suffix}"))
            dst_paths.append(str(Path(self.dst_dir) / f"{Path(src_path).name}"))
        return dst_paths