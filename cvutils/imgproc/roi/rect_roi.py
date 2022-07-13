#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
from typing import List

import cv2
import numpy as np

from ...calib3d.core.core import project_pts
from ...core.core import as_array, check_array, check_array_shape
from .base_roi import ROI, ROIError

__all__ = [
    "RectROI",
]


class RectROI(ROI):
    """Rectangle-like ROI

    A rectangle-like roi is a rotated rect(known as min-area-rect) or
    axis aligned rect(also known as bounding box).

    Args:
        cxcy (list, tuple): center of the rectangle
            Center of rectangle, represented by array_like object with 2 entries.
        wh (list, tuple): width and height of the rectangle
            Size of rectangle, represented by array_like object with 2 entries.
        angle_deg (float): rotation angle in degree of the rectangle
            The rotation angle in a clockwise direction. When the angle is 0, 90,
            180, 270 etc., the rectangle becomes a axis aligned rect, and or it
            becomes a rotated rect.

    Examples:
        python examples/example_rect_roi.py

    """

    def __init__(self, cxcy, wh, angle_deg=0):

        self._cxcy = np.array(cxcy, "float", copy=True).ravel()
        self._wh = np.array(wh, "float", copy=True).ravel()

        check_array_shape(valid_shape=(2,), cxcy=self._cxcy, wh=self._wh)

        self._angle_deg = float(angle_deg)

    def __repr__(self):
        return (
            f"<{self.__class__.__name__}>, "
            f"({self.cxcy}, {self.wh}, {self.angle_deg:.2f})"
        )

    @property
    def cxcy(self):
        """center of the rectangle"""
        return tuple(self._cxcy)

    @property
    def wh(self):
        """width and height of rectangle"""
        return tuple(self._wh)

    @property
    def angle_deg(self):
        """rotated angle of rectangle"""
        return self._angle_deg

    @property
    def area(self):
        """area of rectangle, computed by: w*h"""
        return self.wh[0] * self.wh[1]

    @property
    def perimeter(self):
        """perimeter of rectangle, computed by: 2 * (w + h)"""
        return 2 * (self.wh[0] + self.wh[1])

    @property
    def is_axis_aligned(self) -> bool:
        """Roi is axis aligned or not"""
        return (abs(self.angle_deg) % 90) < 1e-8

    @property
    def axis_aligned_rect(self) -> "RectROI":
        return self.create_from_pts2d(self.vertices_4x2.T, rotated=False)

    @property
    def vertices_4x2(self) -> np.ndarray:
        """Vertices(corners) of the rectangle
        The order is bottomLeft, topLeft, topRight, bottomRight.
        """
        pts_nx2 = cv2.boxPoints((self.cxcy, self.wh, self.angle_deg))

        return np.reshape(pts_nx2, (-1, 2))

    @property
    def bounding_rect(self):
        x, y, w, h = cv2.boundingRect(self.vertices_4x2)
        return x, y, w - 1, h - 1

    @classmethod
    def create_from_pts2d(cls, pts_2xn, rotated=True) -> "RectROI":
        """Create rectangle roi from 2d points set

        When use cv2.minAreaRect (ref: https://blog.csdn.net/lanyuelvyun/article/details/76614872):
        - The angle_deg is calculated by rotating horizontal axis (x axis) counterclockwise
          until it stops on the **first side** of the rectangle.
        - The length of the **first side** is width, the length of another side is height.
        - In OpencV, the origin of the coordinate system is in the upper left corner, the rotation Angle is
          negative counterclockwise. Therefore, θ ∈ (-90, 0].

        Args:
            pts_2xn (2d array_like): 2d points set in shape of 2xn
            rotated (bool): create rotate rect or not
                If set to True, the create rect will be a rotated rect(min area rect),
                or a axis aligned rect(bounding box) rect will be created.

        Returns:
            rect (RectROI): created rectangle

        """
        check_array(shape=(2, -1), pts_2xn=pts_2xn)
        pts_nx2 = np.float32(pts_2xn).T

        if rotated:
            cxcy, wh, angle_deg = cv2.minAreaRect(pts_nx2)
            return cls(cxcy, wh, angle_deg)
        else:
            x, y, w, h = cv2.boundingRect(pts_nx2)
            return cls.create_from_xywh(x, y, w, h)

    @classmethod
    def create_from_xywh(cls, x, y, w, h) -> "RectROI":
        def _to_cxcywh():
            x0, y0, x1, y1 = x, y, x + w, y + h

            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0

            return cx, cy, w, h

        cxcywh = _to_cxcywh()

        return cls(cxcywh[:2], cxcywh[2:], angle_deg=0.0)

    @classmethod
    def create_from_xyxy(cls, x0, y0, x1, y1) -> "RectROI":
        if x0 >= x1:
            raise ValueError(f"'x0' must smaller than 'x1', x0: {x0}, x1: {x1}")
        if y0 >= y1:
            raise ValueError(f"'y0' must smaller than 'y1', y0: {y0}, y1: {y1}")

        w, h = x1 - x0, y1 - y0

        return cls.create_from_xywh(x0, y0, w, h)

    def copy(self) -> "RectROI":
        return RectROI(self._cxcy, self._wh, self._angle_deg)

    def draw(self, img, color, thickness, offset=(0, 0)):
        offset = as_array(offset, dtype="int", shape=2)
        cv2.drawContours(
            img, [np.int0(self.vertices_4x2) + offset], 0, color, thickness
        )

    def transform(
        self,
        translation=(0, 0),
        rotate_angle_deg=0,
        rot_center=None,
        return_trans_M=False,
    ):
        self._angle_deg = self._angle_deg + rotate_angle_deg

        if rot_center is None:
            rot_center = tuple(self._cxcy.tolist())

        matrix = cv2.getRotationMatrix2D(
            center=rot_center, angle=-rotate_angle_deg, scale=1
        )
        matrix[:, 2] += translation
        matrix = np.concatenate([matrix, np.array([[0.0, 0.0, 1.0]])], axis=0)

        cxcy_2x1 = self._cxcy[..., None]
        self._cxcy = project_pts(cxcy_2x1, matrix).ravel()

        if not return_trans_M:
            return None

        return matrix

    def scale(self, fx, fy=None, return_trans_M=False):  # type: ignore
        """Scale(inplace) rectangle

        Args:
            fx (float): a positive number, scaling factor of width
            fy (float): a positive number, scaling factor of height
            return_trans_M (bool): whether to return the transform matrix

        Notes:
            While current RectROI is not axis aligned and fx != fy, the scaled ROI
            is not a rectangle anymore but a parallelogram.
            Tn that case, a minimum rotated rectangle of the parallelogram will
            be made as the scaled result.
        """
        if fx < 0:
            raise ValueError(f"'fx' must be positive, fx: {fx}")

        if fy is None:
            fy = fx
        elif fy < 0:
            raise ValueError(f"'fy' must be positive. fy: {fy}")

        matrix = np.array([[fx, 0, 0], [0, fy, 0], [0, 0, 1]], dtype=np.float32)

        vects = self.vertices_4x2 * np.array([fx, fy], dtype=np.float32)
        cxcy, wh, angle_deg = cv2.minAreaRect(vects)

        self._cxcy = np.array(cxcy, "float").ravel()
        self._wh = np.array(wh, "float").ravel()
        self._angle_deg = float(angle_deg)

        if not return_trans_M:
            return None

        return matrix

    def expand(
        self,
        expand_w_percent=None,
        expand_w_pixel=None,
        expand_h_percent=None,
        expand_h_pixel=None,
        return_trans_M=False,
    ):
        """
        Expand(inplace) the RectROI.

        Args:
            expand_w_percent: expand 'expand_w_percent' * width pixels on each side of width.
                The value must greater than -0.5.
            expand_w_pixel: expand 'expand_w_pixel' pixels on each side of width.
                The value must greater than -0.5 * width.
                'expand_w_percent' and 'expand_w_pixel' could not be None at the same time.
            expand_h_percent: expand 'expand_h_percent' * height pixels on each side of height.
                The value must greater than -0.5.
            expand_h_pixel: expand 'expand_h_pixel' pixels on each side of height.
                The value must greater than -0.5 * height.
                If  'expand_h_percent' and 'expand_h_pixel' are both None: 'expand_h_percent'
                would be set to 'expand_w_percent' if 'expand_w_percent' is not None. Otherwise,
                'expand_h_pixel' would be set to 'expand_w_pixel'
            return_trans_M (bool): whether to return the transform matrix

        Returns:
            matrix(np.ndarray): the transformation matrix
        """
        org_vect = self.vertices_4x2.copy()

        # check expand on x
        if expand_w_percent is not None:
            if expand_w_percent <= -0.5:
                raise ValueError(
                    f"'expand_w_percent' must > -0.5, "
                    f"\nexpand_w_percent: {expand_w_percent}"
                )

            expand_w_pixel = self._wh[0] * expand_w_percent

        elif expand_w_pixel is not None:
            if expand_w_pixel <= -self._wh[0] / 2:
                raise ValueError(
                    f"'expand_w_pixel' must > -1 * half width of ROI, "
                    f"\nexpand_w_pixel: {expand_w_pixel}; half width of ROI: {self._wh[0] / 2}"
                )

        else:
            raise ValueError(
                "'expand_w_pixel' and 'expand_w_percent' could not be None at the same time."
            )

        # check expand on y
        if expand_h_percent is not None:
            if expand_h_percent <= -0.5:
                raise ValueError(
                    f"'expand_h_percent' must > -0.5, "
                    f"\nexpand_h_percent: {expand_h_percent}"
                )
            expand_h_pixel = self._wh[1] * expand_h_percent

        elif expand_h_pixel is not None:
            if expand_h_pixel <= -self._wh[1] / 2:
                raise ValueError(
                    f"'expand_h_pixel' must > -1 * half height of ROI, "
                    f"\nexpand_h_pixel: {expand_h_pixel}; half height of ROI: {self._wh[1] / 2}"
                )

        else:
            expand_h_pixel = (
                expand_w_pixel
                if expand_w_percent is None
                else self._wh[1] * expand_w_percent
            )

        # expand
        self._wh += [2 * expand_w_pixel, 2 * expand_h_pixel]

        if return_trans_M:
            matrix, _ = cv2.findHomography(org_vect, self.vertices_4x2)
        else:
            matrix = None

        return matrix

    def valid_bounding_rect(self, img_size_wh):
        # compute bounding box first
        x, y, w, h = self.bounding_rect

        img_w, img_h = img_size_wh[:2]

        # adjust roi_xywh to fit with image size
        x0 = max(x, 0)
        y0 = max(y, 0)
        x1 = min(x + w, img_w)
        y1 = min(y + h, img_h)

        return x0, y0, x1 - x0, y1 - y0

    def crop(self, img: np.ndarray, copy_=True) -> np.ndarray:
        """Cut roi sub-image from given image

        The bounding box of the roi will be calculated first, and then
        use it to cut sub-image from given image.

        Args:
            img (np.ndarray): source image, 2d or 3d array
            copy_ (bool): copy from source image or not

        Returns:
            np.ndarray: extracted image
        """
        if img.ndim < 2:
            raise ValueError(f"expect ndim > 2 for `img`, while input is {img.ndim}")

        img_h, img_w = img.shape[:2]
        x, y, w, h = self.valid_bounding_rect((img_w, img_h))

        # check adjusted roi
        if (w <= 0) or (h <= 0):
            raise ROIError(f"empty roi {self} on given image!")

        # extract from source image
        roi_on_source_img = img[y : (y + h), x : (x + w)]

        roi_img = np.copy(roi_on_source_img) if copy_ else roi_on_source_img

        return roi_img

    def cut_img(self, img: np.ndarray, copy_=True) -> np.ndarray:

        warnings.warn(
            "\n'cut_img' is deprecated and will be removed in a future version. "
            "Please use the alternative function 'crop'.",
            category=DeprecationWarning,
        )

        roi_img = self.crop(img=img, copy_=copy_)
        return roi_img

    # TODO: more efficient implement
    def contain(self, xy):
        xy = tuple(np.array(xy, "float").ravel().tolist())

        # -1: out, 1: inner, 0: edge
        flag = cv2.pointPolygonTest(self.vertices_4x2, xy, measureDist=False)

        return flag >= 0

    def dist(self, pts_nx2, measure_dist=True) -> List:
        check_array(shape=(-1, 2), pts_nx2=pts_nx2)
        pts = np.array(pts_nx2, "float")

        dist_list = []
        roi_pts_4x2 = self.vertices_4x2

        for xy in pts:
            dist = cv2.pointPolygonTest(
                roi_pts_4x2, tuple(xy), measureDist=measure_dist
            )
            dist_list.append(dist)

        return dist_list
