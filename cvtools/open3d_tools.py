import time

import cv2
import open3d as o3d
import numpy as np
import math

def get_pcd_img_hw(pcd_path):
    img_h = None
    img_w = None
    with open(pcd_path, "rb") as f:
        for i in range(11):
            data = f.readline().decode(encoding="utf-8")
            if data.startswith("HEIGHT"):
                img_h = int(data[6:].strip(" "))
            elif data.startswith("WIDTH"):
                img_w = int(data[5:].strip(" "))
    return img_h, img_w


def get_pcd_panorama(pcd, img_hw, return_img=False):
    panorama = np.array(pcd.colors, dtype=np.float32).reshape((*img_hw, 3))
    if not return_img:
        return panorama
    else:
        return (panorama * 255).astype(np.uint8)


def get_pcd_normals(pcd, img_hw, down_sample=1, return_img=False):
    points = np.array(pcd.points, dtype=np.float32).reshape((*img_hw, 3))
    # pad
    down_points = points[::down_sample, ::down_sample]
    down_hw = down_points.shape[:2]
    down_pcd = o3d.geometry.PointCloud()
    down_pcd.points = o3d.utility.Vector3dVector(down_points.reshape(-1, 3))
    down_pcd.estimate_normals()
    down_pcd.orient_normals_towards_camera_location(camera_location=np.array([0, 0, 0]))
    down_normals = np.array(down_pcd.normals, dtype=np.float32).reshape((*down_hw, 3))

    normals = cv2.resize(down_normals, dsize=(img_hw[1], img_hw[0]), interpolation=cv2.INTER_NEAREST)
    if not return_img:
        return normals
    else:
        return (((normals * 0.5) + 0.5) * 255).astype(np.uint8)


def get_circle_projection_and_depth(pcd, m_per_pixel=0.005, project_depth=None, return_proj_img=True):
    pcd_points = np.array(pcd.points, dtype=np.float32)
    pcd_points /= m_per_pixel
    magnitudes, angle_rads = cv2.cartToPolar(pcd_points[:, 0], pcd_points[:, 1])
    magnitudes = magnitudes[:, 0]
    angle_rads = angle_rads[:, 0]
    project_depth = float(np.mean(magnitudes)) if project_depth is None else project_depth
    img_max_x = int(project_depth * 2 * math.pi)
    z_min = float(np.min(pcd_points[:, -1]))
    z_max = float(np.max(pcd_points[:, -1]))
    img_max_y = math.ceil(z_max - z_min)
    coord_xy = np.stack(((angle_rads * img_max_x / (2 * math.pi) + 0.499).astype(np.int32),
                         (z_max - pcd_points[:, -1] + 0.499).astype(np.int32)), axis=1)
    img_h = img_max_y + 1
    img_w = img_max_x + 1
    pcd_colors = np.array(pcd.colors, dtype=np.float32)
    img_proj = np.zeros((img_h, img_w, 3), dtype=np.float32)
    img_deep = np.zeros((img_h, img_w), dtype=np.float32)
    img_proj[coord_xy[:, 1], coord_xy[:, 0]] = pcd_colors
    img_deep[coord_xy[:, 1], coord_xy[:, 0]] = magnitudes
    if return_proj_img:
        img_proj = (img_proj * 255).astype(np.uint8)
    return img_proj, img_deep

def get_projection(pcd, mask=None, m_per_pixel=0.005, return_proj_img=True):
    pcd_points = np.array(pcd.points, dtype=np.float32)
    pcd_colors = np.array(pcd.colors, dtype=np.float32)
    if mask is None:
        mask = pcd_points[:, 2] > 0
    pcd_points = pcd_points[mask]
    pcd_colors = pcd_colors[mask]
    pcd_points /= m_per_pixel
    x_min = np.min(pcd_points[:, 0])
    x_max = np.max(pcd_points[:, 0])
    y_min = np.min(pcd_points[:, 1])
    y_max = np.max(pcd_points[:, 1])
    img_w = math.ceil(x_max - x_min) + 1
    img_h = math.ceil(y_max - y_min) + 1
    coord_xy = np.stack(
        ((pcd_points[:, 0] - x_min + 0.4999).astype(np.int32), (pcd_points[:, 1] - y_min + 0.4999).astype(np.int32)),
        axis=-1,
    )
    img = np.zeros((img_h, img_w, 3), dtype=np.float32)
    img[coord_xy[:, 1], coord_xy[:, 0]] = pcd_colors
    if return_proj_img:
        img = (img * 255).astype(np.uint8)
    return img

def get_rotation_matrix_from_normals(normals):
    normals = normals / (normals ** 2).sum() ** 0.5
    axis_angle = np.arccos(normals)
    print(axis_angle)
    rm = o3d.geometry.PointCloud.get_rotation_matrix_from_axis_angle(axis_angle)
    return rm



