import torch
import cv2
import numpy as np
import torch.nn.functional as F

def preprocess(
    img,
    input_shape_hw=(544, 1120),
    img_mean=(123.675, 116.28, 103.53),
    img_std=(58.395, 57.12, 57.375),
    to_rgb=True,
):
    img_resize = cv2.resize(img, (input_shape_hw[1], input_shape_hw[0]))
    if to_rgb:
        cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
    img_resize = img_resize.astype(np.float64)
    for i, (m, s) in enumerate(zip(img_mean, img_std)):
        img_resize[:, :, i] = (img_resize[:, :, i] - m) / s

    return img_resize

def infer_by_libtorch_module(imgs, model, device, **kwargs):
    single = False
    if isinstance(imgs, np.ndarray):
        imgs = [imgs]
        single = True
    imgs_preprocess = [preprocess(img, **kwargs)for img in imgs]

    # single image to tensor
    # imgs_preprocess_tensor = torch.tensor(imgs_preprocess, dtype=torch.float32).permute(0, 3, 1, 2)
    imgs_preprocess_tensor = torch.from_numpy(np.array(imgs_preprocess, dtype=np.float32).transpose((0, 3, 1, 2)))

    # to device
    imgs_preprocess_tensor = imgs_preprocess_tensor.to(device)

    # infer to get results
    model.eval().to(device)
    results = model(imgs_preprocess_tensor)

    results_format = []

    for img_id, result in enumerate(results):
        scores, masks, labels = result
        masks = masks.to(torch.float32)
        masks = F.interpolate(masks.unsqueeze(0), size=imgs[img_id].shape[:2], align_corners=False, mode='bilinear')[0]
        masks = masks > 0.5
        masks = list(masks.detach().cpu().numpy())
        labels = list(labels.detach().cpu().numpy())
        scores = list(scores.detach().cpu().numpy())
        results_format.append((masks, labels, scores))
    if single:
        results_format = results_format[0]
    return results_format
