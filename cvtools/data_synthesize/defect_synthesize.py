import math
import random
from typing import Union

import cv2
import numpy as np
from rrcvutils import PointSet, PolygonROI

__all__ = [
    "BaseDefectSynthesizer",
    "gen_mask_by_point",
    "gen_mask_by_polygon",
    "gen_mask_by_broken_line",
    "gen_mask_by_bezier_curve",
    "gen_color_by_random_scale",
    "gen_color_by_random_noise",
]


class BaseDefectSynthesizer(object):
    def __init__(
        self,
    ):
        pass

    def run(
        self,
        image: np.ndarray,
        mask: Union[np.ndarray, None] = None,
        allowed_mask: Union[np.ndarray, None] = None,
        run_times: int = 1,
        is_overlap: bool = False,
        use_blur: bool = False,
        blur_keneral_size: int = 3,
        blur_sigma: float = 7,
        blur_padding_size: int = 7,
    ):
        assert image.ndim == 2
        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        if allowed_mask is None:
            allowed_mask = np.ones_like(mask, dtype=np.bool_)
        else:
            allowed_mask = allowed_mask > 0
        for i in range(run_times):
            defect_mask = self.gen_mask()
            # 需要保证：图像中缺陷与图像边界接壤，否则腐蚀可能会提供让缺陷粘贴出界的落脚点
            x_ = np.nonzero(np.sum(defect_mask, axis=0))[0]
            y_ = np.nonzero(np.sum(defect_mask, axis=1))[0]
            defect_mask = defect_mask[min(y_) : max(y_) + 1, min(x_) : max(x_) + 1]

            # 缺陷mask太大，无法粘贴则跳出
            if (
                defect_mask.shape[0] > mask.shape[0]
                or defect_mask.shape[1] > mask.shape[1]
            ):
                continue

            if not is_overlap:
                allowed_mask = np.bitwise_and(allowed_mask, mask == 0)
            defect_allowed_mask = (
                np.pad(
                    allowed_mask,
                    pad_width=(
                        (defect_mask.shape[0], defect_mask.shape[0]),
                        (defect_mask.shape[1], defect_mask.shape[1]),
                    ),
                    constant_values=0,
                ).astype(np.uint8)
                * 255
            )

            defect_allowed_mask = cv2.erode(
                src=defect_allowed_mask,
                kernel=defect_mask,
                anchor=(defect_mask.shape[1] // 2, defect_mask.shape[0] // 2),
            )
            defect_allowed_mask = defect_allowed_mask[
                defect_mask.shape[0] : defect_allowed_mask.shape[0]
                - defect_mask.shape[0],
                defect_mask.shape[1] : defect_allowed_mask.shape[1]
                - defect_mask.shape[1],
            ]

            rows, cols = np.nonzero(defect_allowed_mask)
            # 无法粘贴则跳出
            if len(rows) == 0:
                continue
            pt_id = random.randint(0, len(rows) - 1)
            cy = rows[pt_id]
            cx = cols[pt_id]
            dst_row_0 = cy - defect_mask.shape[0] // 2
            dst_col_0 = cx - defect_mask.shape[1] // 2
            dst_row_1 = dst_row_0 + defect_mask.shape[0]
            dst_col_1 = dst_col_0 + defect_mask.shape[1]

            # 粘贴
            image_patch = image[dst_row_0:dst_row_1, dst_col_0:dst_col_1]
            image[dst_row_0:dst_row_1, dst_col_0:dst_col_1] = image_patch * (
                defect_mask == 0
            ) + self.color_mask(
                image=image_patch,
                mask=defect_mask,
            )
            mask_patch = mask[dst_row_0:dst_row_1, dst_col_0:dst_col_1]
            mask[dst_row_0:dst_row_1, dst_col_0:dst_col_1] = (
                mask_patch * (defect_mask == 0) + defect_mask
            )

            if use_blur:
                assert blur_keneral_size & 1 == 1 and blur_padding_size & 1 == 1
                pad_width = (
                    (
                        min(blur_padding_size, dst_row_0),
                        min(blur_padding_size, image.shape[0] - dst_row_1),
                    ),
                    (
                        min(blur_padding_size, dst_col_0),
                        min(blur_padding_size, image.shape[1] - dst_col_1),
                    ),
                )

                image_patch = image[
                    dst_row_0 - pad_width[0][0] : dst_row_1 + pad_width[0][1],
                    dst_col_0 - pad_width[1][0] : dst_col_1 + pad_width[1][1],
                ]
                circle_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, ksize=(blur_keneral_size, blur_keneral_size)
                )

                # image_patch_blur = cv2.filter2D(image_patch, -1, circle_kernel / np.sum(circle_kernel))
                image_patch_blur = cv2.GaussianBlur(
                    image_patch,
                    ksize=(blur_keneral_size, blur_keneral_size),
                    sigmaX=blur_sigma,
                )
                mask_patch = np.pad(defect_mask > 0, pad_width=pad_width).astype(
                    np.uint8
                )
                mask_patch_dilate = cv2.dilate(mask_patch, circle_kernel)
                image[
                    dst_row_0 - pad_width[0][0] : dst_row_1 + pad_width[0][1],
                    dst_col_0 - pad_width[1][0] : dst_col_1 + pad_width[1][1],
                ] = image_patch * (mask_patch_dilate == 0) + image_patch_blur * (
                    mask_patch_dilate > 0
                )

        return image, mask

    def gen_mask(self):
        raise NotImplementedError

    def color_mask(self, image, mask):
        raise NotImplementedError


def gen_mask_by_point(
    radius_range: Union[tuple, list],
):
    radius = random.randint(*radius_range)
    d = radius * 2 + 1
    mask = np.zeros((d, d), dtype=np.uint8)
    cv2.circle(
        img=mask, center=(radius, radius), radius=radius, thickness=-1, color=(255,)
    )
    return mask


def gen_mask_by_polygon(
    num_points_range: Union[tuple, list],
    area_range: Union[tuple, list],
    return_convex_hull: bool = False,
    refine: bool = False,
):

    # refine：对mask进行resize，获取更接近设定面积大小的mask
    assert min(num_points_range) > 2
    num_points = random.randint(*num_points_range)
    dst_area = random.randint(*area_range)

    points = np.random.random((num_points, 2)).astype(np.float32)
    if return_convex_hull:
        points = cv2.convexHull(points)[:, 0]
    point_set = PointSet(points)
    point_set.sort()

    src_area = cv2.contourArea((point_set.pts_nx2).astype(np.float32), oriented=False)
    scale_factor = (dst_area / src_area) ** 0.5
    point_set.scale(scale_factor, scale_factor)

    points = point_set.pts_nx2
    min_x, min_y = np.min(points[:, 0]), np.min(points[:, 1])
    points[:, 0] -= min_x
    points[:, 1] -= min_y
    max_x, max_y = np.max(points[:, 0]), np.max(points[:, 1])

    mask = np.zeros((math.ceil(max_y), math.ceil(max_x)), dtype=np.uint8)
    polygon = PolygonROI(points)
    polygon.draw(
        img=mask,
        color=(255,),
        thickness=-1,
    )

    if refine:
        src_area = np.count_nonzero(mask)
        scale_factor = dst_area / src_area
        mask = cv2.resize(
            mask, dsize=None, fx=scale_factor**0.5, fy=scale_factor**0.5
        )
        _, mask = cv2.threshold(mask, thresh=127, maxval=255, type=cv2.THRESH_BINARY)

    return mask


def gen_mask_by_bezier_curve(
    curve_order_range: Union[tuple, list],
    length_range: Union[tuple, list],
    thickness_range: Union[tuple, list],
    fine_error: float = 0.01,
    is_order: bool = False,
    is_closed: bool = False,
):
    assert curve_order_range[0] >= 3
    points_num = random.randint(*curve_order_range)
    points = np.random.random((points_num, 2)).astype(np.float32)
    # length = random.randint(*length_range)
    length = random.random() * (length_range[1] - length_range[0]) + length_range[0]
    thickness = random.randint(*thickness_range)

    if is_order:
        point_set = PointSet(points)
        point_set.sort()
        points = point_set.pts_nx2
        i = random.randint(0, points_num)
        points = np.vstack([points[i:], points[:i]])

    curve_points = gen_bezier_curve(points, fine_error=fine_error)
    src_length = cv2.arcLength(curve_points, closed=False)
    scale_factor = length / src_length
    curve_points = curve_points * scale_factor

    min_x, min_y = np.min(curve_points[:, 0]), np.min(curve_points[:, 1])
    curve_points[:, 0] -= min_x
    curve_points[:, 1] -= min_y
    max_x, max_y = np.max(curve_points[:, 0]), np.max(curve_points[:, 1])

    mask = np.zeros(
        (math.ceil(max_y) + thickness * 2, math.ceil(max_x) + thickness * 2),
        dtype=np.uint8,
    )
    point_set = PointSet((curve_points + thickness + 0.5).astype(np.int32))
    point_set.draw(
        mask, color=(255,), thickness=thickness, line_cfg=dict(is_closed=is_closed)
    )

    return mask


def gen_mask_by_broken_line(
    points_num_range: Union[tuple, list],
    length_range: Union[tuple, list],
    angle_deg_range: Union[tuple, list],
    thickness_range: Union[tuple, list],
    start_angle_deg_range: Union[tuple, list] = (0, 0),
    is_closed: bool = False,
):
    points_num = random.randint(*points_num_range)
    thickness = random.randint(*thickness_range)
    cur_angle = (
        random.random() * (start_angle_deg_range[1] - start_angle_deg_range[0])
        + start_angle_deg_range[0]
    )
    point_list = [(0, 0)]

    for i in range(points_num - 1):
        length = random.randint(*length_range)
        angle = (
            random.random() * (angle_deg_range[1] - angle_deg_range[0])
            + angle_deg_range[0]
            + cur_angle
        )
        cur_angle = angle

        cur_point = point_list[-1]
        next_point = (
            round(length * math.cos(math.radians(angle))) + cur_point[0],
            round(length * math.sin(math.radians(angle))) + cur_point[1],
        )
        point_list.append(next_point)

    points = np.array(point_list, dtype=np.float32)
    min_x, min_y = np.min(points[:, 0]), np.min(points[:, 1])
    points[:, 0] -= min_x
    points[:, 1] -= min_y
    max_x, max_y = np.max(points[:, 0]), np.max(points[:, 1])

    mask = np.zeros(
        (math.ceil(max_y) + thickness * 2, math.ceil(max_x) + thickness * 2),
        dtype=np.uint8,
    )
    point_set = PointSet((points + thickness + 0.5).astype(np.int32))
    point_set.draw(
        mask,
        color=(255,),
        thickness=1,
        radius=thickness,
        line_cfg=dict(is_closed=is_closed),
    )
    return mask


def gen_color_by_random_noise(
    mask: np.ndarray,
    clip_range: Union[tuple, list] = (0, 255),
):
    min_ = max(clip_range[0], 0)
    max_ = min(clip_range[1], 255)

    noise = np.random.randint(min_, max_, mask.shape, dtype=np.uint8)
    return (noise * (mask > 0)).astype(np.uint8)


def gen_color_by_random_scale(
    mask: np.ndarray,
    clip_range: Union[tuple, list] = (0, 255),
    image: Union[np.ndarray, None] = None,
    scale_range: Union[tuple, list] = (1, 1),
    offset_range: Union[tuple, list] = (0, 0),
    apply_pixel: bool = False,
):
    if image is None:
        image = np.zeros_like(mask, dtype=np.uint8)
    else:
        assert image.ndim == 2
        assert image.shape[:2] == mask.shape
    min_ = max(clip_range[0], 0)
    max_ = min(clip_range[1], 255)

    if apply_pixel:
        scale = scale_range[0] + np.random.random(image.shape) * (
            scale_range[1] - scale_range[0]
        )
        offset = offset_range[0] + np.random.random(image.shape) * (
            offset_range[1] - offset_range[0]
        )
    else:
        scale = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
        offset = offset_range[0] + random.random() * (offset_range[1] - offset_range[0])

    image_trans = (image.astype(np.float32) * scale + offset).clip(min_, max_)

    return (image_trans * (mask > 0)).astype(np.uint8)


def calNextPoints(points, rate):
    if points.shape[0] == 1:
        return points
    left = points[:-1]
    right = points[1:]
    ans = (right - left) * rate + left
    return calNextPoints(ans, rate)


def gen_bezier_curve(points, fine_error=0.01):
    new_points = []
    for r in np.arange(0, 1, fine_error):
        next_point = calNextPoints(points, rate=r)
        new_points.append(next_point)
    new_points = np.array(new_points)[:, 0]
    return new_points
