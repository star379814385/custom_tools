from custom_tools.aitools.labelme_tools.labelme_dataset import LabelmeDataset

data_dir = r"D:\dataset\KolektorSDD_pos"
labelme_dataset = LabelmeDataset.build_from_seg_mask(
    data_dir=data_dir,
    img_pattern="*.jpg",
    label_fn=lambda p: p.replace(".jpg", "_label.bmp"),
    color2name={
        (255, 255, 255): "defect",
    }
)
labelme_dataset.save()