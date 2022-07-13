#!/usr/bin/env python
# -*- coding:utf-8 -*-


import math
from typing import List, Tuple, Union

import cv2
import numpy as np

from .drawing import draw_line

__all__ = [
    "gamma_transform",
    "contrast_stretch",
    "laplacian",
    "find_max_area_contour_index",
    "find_topk_area_contours_indices",
    # "get_hist_img",
    "rotate_img",
    "pad_img",
    "resize_img",
    "find_contours",
    "binary_otsu",
    "roundness",
]


def _raise(msg):
    raise Exception(msg)
    # raise RRCVException(msg)


def _gamma_transform(src, gamma):
    lookup_table = []
    constant = 255.0 / (255**gamma)
    for i in range(256):
        lookup_table.append(constant * (i**gamma))

    return cv2.LUT(src, np.array(lookup_table, dtype=np.uint8))


def gamma_transform(img: np.ndarray, gamma) -> np.ndarray:
    if img.ndim not in (2, 3):
        _raise("ndim must be 2 or 3")

    if img.ndim == 2:
        return _gamma_transform(img, gamma)

    hsv_img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2HSV)
    h = hsv_img[:, :, 0]
    s = hsv_img[:, :, 1]
    v = hsv_img[:, :, 2]
    v = _gamma_transform(v, gamma)
    new_hsv_img = cv2.merge((h, s, v))

    return cv2.cvtColor(src=new_hsv_img, code=cv2.COLOR_HSV2BGR)


def contrast_stretch(gray: np.ndarray, min_val=0):
    hist = cv2.calcHist([gray], [0], None, [256], [0.0, 256.0])

    idx_min = 0
    for i in range(hist.shape[0]):
        if hist[i] > min_val:
            idx_min = i
            break

    idx_max = 0
    for i in range(hist.shape[0]):
        if hist[255 - i] > min_val:
            idx_max = 255 - i
            break

    _, gray = cv2.threshold(gray, idx_max, idx_max, cv2.THRESH_TRUNC)
    gray = ((gray >= idx_min) * gray) + ((gray < idx_min) * idx_min)
    res = np.uint8(255.0 * (gray - idx_min) / (idx_max - idx_min))

    return res


def laplacian(img: np.ndarray) -> np.ndarray:
    img_dtype_dict = {
        2: cv2.CV_8UC1,
        3: cv2.CV_8UC3,
    }

    if img.ndim not in img_dtype_dict:
        _raise(f"except ndim of {img_dtype_dict.keys()} array for `img`")

    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    enhanced_img = cv2.filter2D(img, img_dtype_dict[img.ndim], kernel)

    return enhanced_img


def find_topk_area_contours_indices(contours, num=1) -> np.ndarray:
    """Find indices of contours with top-n area

    Args:
        contours(List[array_like]): List of ndarray, comes from OpenCV.
        num(int): Number indicating the number of top-n, default is 1.

    Returns:
        np.ndarray: The indices of top-n area contours.
    """
    if not isinstance(num, int):
        _raise("var_value must be int, not {}".format(type(num)))
    if num <= 0:
        _raise("var_value must be greater than 0: var_value={}".format(num))
    if len(contours) < num:
        _raise("contours var_value is less than %s" % (num,))

    contours_area = [cv2.contourArea(ctn) for ctn in contours]
    increasing_sorted_idx = np.argsort(contours_area)
    decreasing_sorted_idx = increasing_sorted_idx[::-1]
    topn_indices = decreasing_sorted_idx[:num]

    return topn_indices


def find_max_area_contour_index(contours) -> int:
    """Find index of contour with max area"""
    return find_topk_area_contours_indices(contours, num=1)[0]


# def get_hist_img(
#     img_or_hist, channels=0, mask=None, hist_size=256, ranges=(0.0, 256.0)
# ):
#     if img_or_hist.shape[1] != 1:
#         assert img_or_hist.ndim == 2
#         hist = cv2.calcHist(
#             images=[img_or_hist],
#             channels=[channels],
#             mask=mask,
#             histSize=[hist_size],
#             ranges=ranges,
#         )
#     else:
#         hist = img_or_hist
#
#     hist = as_array(hist, "float32", shape=-1)
#     max_val = float(np.max(hist))
#
#     # init show hist image
#     hist_img = np.zeros((hist.shape[0], hist.shape[0]))
#     hist_pt = hist.shape[0] * 0.9 / max_val
#     for i, value in enumerate(hist):
#         draw_line(hist_img, (i, hist_size - value * hist_pt), (i, 0), 255, 1)
#
#     return hist_img


def rotate_img(
    img: np.ndarray, angle_deg, center_xy=None, interpolation=cv2.INTER_NEAREST
):
    """Rotate an image with specified point and angle.

    Args:
        img(array_like): A ndarray.
        angle_deg(int): rotation angle.
        center_xy(tuple): Center of rotation.
        interpolation(int): Interpolation method.

    Returns:
        np.ndarray: The indices of top-n area contours.
    """

    w1 = math.fabs(img.shape[1] * math.cos(np.deg2rad(angle_deg)))
    w2 = math.fabs(img.shape[0] * math.sin(np.deg2rad(angle_deg)))
    h1 = math.fabs(img.shape[1] * math.sin(np.deg2rad(angle_deg)))
    h2 = math.fabs(img.shape[0] * math.cos(np.deg2rad(angle_deg)))
    width = int(w1 + w2) + 1
    height = int(h1 + h2) + 1
    dst_size = (width, height)
    x = img.shape[1]
    y = img.shape[0]
    if center_xy is None:
        center = np.array([x / 2, y / 2]).reshape(2, 1)
    else:
        center = center_xy
    rotate_matrix = cv2.getRotationMatrix2D(center=(0, 0), angle=angle_deg, scale=1.0)
    rotate_center = np.dot(rotate_matrix[0:2, 0:2].reshape(2, 2), center)
    rotate_matrix[0, 2] = width / 2 - rotate_center[0]
    rotate_matrix[1, 2] = height / 2 - rotate_center[1]
    if angle_deg % 90 == 0:
        rot90_k = int(angle_deg / 90 % 4)
        rotated_img = np.ascontiguousarray(np.rot90(img, rot90_k))
    else:
        rotated_img = cv2.warpAffine(
            src=img, M=rotate_matrix, dsize=dst_size, flags=interpolation
        )
    transform_matrix = np.vstack((rotate_matrix, np.array([0.0, 0.0, 1.0])))

    return rotated_img, transform_matrix


def pad_img(
    img: np.ndarray,
    dsize_wh: Union[Tuple[int, int], None] = None,
    pad_tblr=(0, 0, 0, 0),
    border_type=cv2.BORDER_CONSTANT,
    fill_value: Union[Tuple[int, int, int], int] = (255, 255, 255),
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Pad an image with `border_type`. If `border_type` is cv2.BORDER_CONSTANT, padded area will
    be filled by `fill_value`.
        If `dsize_wh` is not None, img will be centered on padded_img, otherwise img will be padded
    by `pad_tblr`.

    Args:
        img(array_like): A ndarray.
        dsize_wh(tuple): Image size after padded.
        pad_tblr(tuple): The area of the top, bottom, left, right to the source image.
        border_type(int): Interpolation method, Refers to doc of opencv, `BorderTypes` for more details.
        fill_value(tuple): Fill the value when border_type is cv2.BORDER_CONSTANT

    Returns:
        np.ndarray: The padded image.
        tuple: The area of the top, bottom, left, right to the source image.
    """

    if dsize_wh is not None:
        src_h, src_w = img.shape[:2]
        dst_w, dst_h = dsize_wh

        pad_l = int((dst_w - src_w) / 2)
        pad_r = dst_w - src_w - pad_l
        pad_t = int((dst_h - src_h) / 2)
        pad_b = dst_h - src_h - pad_t
        pad_tblr = (pad_t, pad_b, pad_l, pad_r)
    else:
        pad_t, pad_b, pad_l, pad_r = pad_tblr

    padded_img = cv2.copyMakeBorder(
        img,
        top=pad_t,
        bottom=pad_b,
        left=pad_l,
        right=pad_r,
        borderType=border_type,
        value=fill_value,
    )
    return padded_img, pad_tblr


def resize_img(
    img, dsize_wh, keep_aspect_ratio=False, interpolation=cv2.INTER_NEAREST
) -> np.ndarray:
    """Resize an image into `dsize`.

    Args:
        img(array_like): A ndarray.
        dsize_wh(tuple): Image size after resize.
        keep_aspect_ratio(bool): Whether keep the ratio of width and height.
        interpolation(int): Interpolation method of resize.

    Returns:
        np.ndarray: The resized image.

    """

    if not keep_aspect_ratio:
        resized_img = cv2.resize(img, dsize_wh, interpolation=interpolation)
    else:
        img_h, img_w = img.shape[:2]
        scale = min(dsize_wh[1] / img_h, dsize_wh[0] / img_w)
        resized_img = cv2.resize(
            img, None, fx=scale, fy=scale, interpolation=interpolation
        )
    return resized_img


def find_contours(
    img, mode, method=cv2.CHAIN_APPROX_NONE, offset=(0, 0)
) -> Tuple[List[np.ndarray], np.ndarray]:
    """A convenient function for compatibility difference opencv version

    Finds contours in a binary image. Refers to doc of opencv.findContours for more details

    Returns:
         Tuple[list, np.ndarray]:
            - list of found contours, a contour is a np.ndarray with shape nx1x2
            - A array containing information about the image topology, nx4

    Raises:
         RuntimeError: If can not determine version of cv2

    """
    ret_vars = cv2.findContours(img, mode, method, offset=offset)

    if len(ret_vars) == 2:
        contours, hierarchy_1xmx4 = ret_vars
    elif len(ret_vars) == 3:
        image, contours, hierarchy_1xmx4 = ret_vars
    else:
        raise RuntimeError("un-known version of cv2")

    # no contours were found, return directly
    if not contours:
        return contours, hierarchy_1xmx4

    hierarchy_nx4 = np.reshape(hierarchy_1xmx4, (-1, 4))

    return contours, hierarchy_nx4


def binary_otsu(gray_img: np.ndarray, inverse=False) -> Tuple[int, np.ndarray]:
    """Binarization given image by OTSU algorithm

    Args:
        gray_img(np.ndarray): grayscale image with dtype='uint8'
        inverse(bool): inverse or not(default)

    Returns:
        Tuple[int, np.ndarray]:
            - threshold value computed by OTSU
            - binary image

    References:
        1. https://en.wikipedia.org/wiki/Otsu%27s_method

    Raises:
        ValueError: If invalid `img` is given

    """
    flag = cv2.THRESH_OTSU | (cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY)

    thresh, binary_img = cv2.threshold(gray_img, 0, 255, flag)

    return thresh, binary_img


def roundness(contour_nx2) -> float:
    """Compute roundness of a closed contour

    For a circle with radius r, it's area computed by: A = pi * r * r,
    and perimeter computed by: L = 2 * pi * r. The roundness of a contour
    is defined as: r = (4 * pi * A) / (L * L). The roundness is 1 for
    a circle, and the rounder the contour, the higher the value.

    Notes:
        Since there is not method to check whether the input contour is closed
        or not, the user should guarantee this requirement!

    Returns:
        float: roundness of contour
    """
    contour_nx2 = as_array(contour_nx2, "float32")
    check_array(shape=(-1, 2), contour_nx2=contour_nx2)

    A = cv2.contourArea(contour_nx2, oriented=False)
    L = cv2.arcLength(contour_nx2, closed=True)

    if abs(L) < 1e-8:
        return 0

    r = (4 * np.pi * A) / (L * L)

    return r
