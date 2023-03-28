from custom_tools.aitools.cvat_tools.core import Cvat

src_dir = r"D:\project\6_qiangban_qiti\dataset\images\墙板（未标注）\A0007-000442 中铁城建西客站片区腊山河西侧3地块项目"
dst_dir = r"D:\project\6_qiangban_qiti\dataset\images\墙板（未标注）\A0007-000442"
cvat = Cvat(
    src_dir=src_dir,
    dst_dir=dst_dir
)
# cvat.copy_src_to_dst()
cvat.copy_dst_labelme_to_src()