#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

# from ..core.core import check_array

__all__ = [
    "draw_pts",
    "draw_line",
    "draw_lines",
    "draw_contours",
    "draw_epilines",
]


def draw_pts(
    img: np.ndarray, pts_2xn, color, radius=1, thickness=-1, offset=(0, 0), shift=0
):

    # check_array(shape=(2, -1), pts_2xn=pts_2xn)
    assert (
        isinstance(pts_2xn, np.ndarray) and pts_2xn.ndim == 2 and pts_2xn.shape[0] == 2
    )

    offset = np.array(offset).ravel()

    for idx in range(pts_2xn.shape[1]):
        pt = int(pts_2xn[0, idx] + offset[0]), int(pts_2xn[1, idx] + offset[1])
        cv2.circle(img, pt, radius, color, thickness, shift=shift)


def draw_lines(
    img,
    pts_nx2,
    color,
    is_closed=False,
    thickness=2,
    offset=(0, 0),
    shift=0,
    line_type=cv2.LINE_8,
):
    offset = np.array(offset).ravel()
    pts_nx2 = (np.reshape(pts_nx2, (-1, 2)) + offset).astype(np.int32)
    pts_nx1x2 = pts_nx2.reshape((-1, 1, 2))

    cv2.polylines(
        img=img,
        pts=[pts_nx1x2],
        isClosed=is_closed,
        color=color,
        thickness=thickness,
        lineType=line_type,
        shift=shift,
    )


def draw_line(img: np.ndarray, pt1, pt2, color, thickness=2, shift=0):
    pt1 = np.round(np.ravel(pt1)).astype("int")
    pt2 = np.round(np.ravel(pt2)).astype("int")

    cv2.line(
        img=img,
        pt1=tuple(pt1),
        pt2=tuple(pt2),
        color=color,
        thickness=thickness,
        shift=shift,
    )


def draw_contours(
    img: np.ndarray,
    contours,
    contour_idx,
    color,
    thickness=1,
    lineType=8,
    hierarchy=None,
    maxLevel=1 << 31 - 1,
    offset=(0, 0),
):
    if hierarchy is not None:
        cv2.drawContours(
            image=img,
            contours=contours,
            contourIdx=contour_idx,
            color=color,
            thickness=thickness,
            lineType=lineType,
            hierarchy=hierarchy,
            maxLevel=maxLevel,
            offset=offset,
        )
    else:
        cv2.drawContours(
            image=img,
            contours=contours,
            contourIdx=contour_idx,
            color=color,
            thickness=thickness,
            lineType=lineType,
            offset=offset,
        )


def draw_epilines(img: np.ndarray, epilines_nx3, color, thickness=3):
    h, w = img.shape[:2]

    min_x, max_x = 0, w - 1

    # ax + by + c = 0 == > y = -(c + ax) / b
    for (a, b, c) in epilines_nx3:
        p0 = (min_x, int(-(c + a * min_x) / b))
        p1 = (max_x, int(-(c + a * max_x) / b))

        cv2.line(img, p0, p1, tuple(color), thickness)
