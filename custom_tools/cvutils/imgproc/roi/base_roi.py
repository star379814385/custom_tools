#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc

import numpy as np

__all__ = [
    "ROI",
    "cvt_roi",
    "ROI_CVT_XYWH2XYXY",
    "ROI_CVT_XYXY2XYWH",
]


class ROI(abc.ABC):
    """Base class for ROI classes"""

    @property
    @abc.abstractmethod
    def area(self):
        """Area of roi"""
        pass

    @property
    @abc.abstractmethod
    def perimeter(self):
        """Perimeter of roi"""
        pass

    @property
    @abc.abstractmethod
    def bounding_rect(self):
        """Bounding rect region of ROI"""
        pass

    @abc.abstractmethod
    def copy(self):
        pass

    @abc.abstractmethod
    def draw(self, img, color, thickness, offset=(0, 0)):
        """Draw ROI outlines or filled.

        Args:
            img (np.ndarray): drawn image
            color (tuple): drawn color in format of BGR and ranged in [0, 255]
            thickness (int): thickness of drawn lines
                If `thickness >= 0`, draws the outlines of rectangle to `img`, or
                fills the area bounded by the rectangle.
            offset(array_like): offset of draw roi, default if (0,0)
        """
        pass

    @abc.abstractmethod
    def scale(self, *args, **kwargs):
        """Scale(inplace) roi"""
        pass

    @abc.abstractmethod
    def expand(self, *args, **kwargs):
        """

        Expand(inplace) the ROI
            - param > 0: expand
            - param == 0: no change
            - param < 0: reduce
        """
        pass

    @abc.abstractmethod
    def transform(
        self, translation, rotate_angle_deg, rot_center=None, return_trans_M=False
    ):
        """Transform(inplace) roi by rotation and translation on 2d image plane

        Args:
            translation (tuple, list, array_like): translation component
                Translation component represented by array_like object with 2 entries

            rotate_angle_deg (float): rotation component
                Rotation component represented by angle of rotation in degree.
                A positive value means rotate clockwise while a negative
                one means rotate counterclockwise.

            rot_center (tuple): rotation center.
                If is None, it will be set to the center of ROI.

            return_trans_M (bool): whether to return the transform matrix

        Notes:
            The coordinate system frame of image plane built like this:
            the origin is the left-top corner of image, and x-axis directs
            right while y-axis directs down.
        """
        pass

    @abc.abstractmethod
    def contain(self, xy):
        """Determine rectangle contains(include coincides) point `xy` or not"""
        pass

    @abc.abstractmethod
    def dist(self, pts_nx2: np.ndarray, measure_dist=True):
        """

        Calculate distances from points to the ROI.

        Args:
            pts_nx2 (np.ndarray): Points tested against the ROI.
            measure_dist (bool): If true, the function estimates the signed distance from the point
                to the nearest contour edge. Otherwise, the function only checks if the point is
                inside a contour or not:
                - -1: Out of ROI.
                - 0:  Inside ROI.
                - 1:  On the ROI.

        Returns:

            dists (list): List of the distance.

        """
        pass

    def mask(
        self, mask_size_wh, mask_value=255, background_value=0, offset=(0, 0)
    ) -> np.ndarray:
        """Return a 'uint8' mask with specified shape

        Args:
            mask_size_wh(array_like): shape of returned mask
            mask_value(int, [0, 255]): value of mask(foreground)
            background_value(int, [0, 255]): value of background
            offset(array_like): offset of mask

        Returns:
            np.ndarray: mask array, with dtype='uint8'

        """
        if not (0 <= mask_value <= 255):
            raise ValueError("expect value in rang of [0, 255] for `mask_value`")
        if not (0 <= background_value <= 255):
            raise ValueError("expect value in rang of [0, 255] for `background_value`")

        w, h = tuple(mask_size_wh)
        mask = np.full((h, w), background_value, "uint8")

        self.draw(mask, (mask_value,), -1, offset)

        return mask

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
    def valid_bounding_rect(self, img_size_wh):
        """Return valid bounding rect region of roi

        Args:
            img_size_wh: image size of wh

        Returns:
             Tuple[int]: [x, y, w, h] format
        """
        pass


ROI_CVT_XYXY2XYWH = 0
ROI_CVT_XYWH2XYXY = 8


def cvt_roi(roi, flag):
    """
    convert roi type

    :param roi: list or ndarray
    :param flag: ROI_CVT_XYXY2XYWH or ROI_CVT_XYWH2XYXY
    :return: roi (xyxy or xywh,depends on what you set)
    """
    x0, y0, c, d = np.array(roi).ravel()
    if ROI_CVT_XYWH2XYXY == flag:
        x1 = x0 + c
        y1 = y0 + d
        return np.array([x0, y0, x1, y1])
    if ROI_CVT_XYXY2XYWH == flag:
        w = c - x0
        h = d - y0
        return np.array([x0, y0, w, h])
    raise ValueError("flag is wrong!!!")
