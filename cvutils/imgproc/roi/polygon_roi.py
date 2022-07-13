from typing import List

import cv2
import numpy as np

from ...calib3d.core.core import project_pts
from ...core.core import as_array, check_array, check_array_shape
from ..core import roundness
from ..drawing import draw_contours
from .base_roi import ROI, ROIError

__all__ = ["PolygonROI"]


class PolygonROI(ROI):
    def __init__(self, pts_nx2):
        pts = as_array(pts_nx2)
        check_array_shape(valid_shape=(-1, 2), pts_nx2=pts)

        self._pts_nx2 = pts

    @property
    def cxcy(self) -> tuple:
        """center of the points in contour array"""
        cx, cy = np.mean(self._pts_nx2, axis=0).tolist()
        return cx, cy

    @property
    def mass_cxcy(self) -> tuple:
        """mass center"""
        moments = cv2.moments(self._pts_nx2)
        if moments["m00"] != 0:
            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])
        else:
            cx, cy = self.cxcy
        return cx, cy

    @property
    def area(self) -> float:
        """Calculate the area of the contour use opencv."""
        return cv2.contourArea(self._pts_nx2.astype(np.float32))

    @property
    def pixel_area(self) -> int:
        """
        Draw the contour on a mask, count the pixels in the contour as the area.
        The result is influenced by 'lineType'
        """

        if self._pts_nx2.shape[0] <= 2:
            return 0

        cnt_max_x, cnt_max_y = np.max(self._pts_nx2, axis=0)
        img = np.zeros((int(cnt_max_y) + 1, int(cnt_max_x) + 1)).astype(np.uint8)
        draw_contours(
            img,
            [np.int0(self._pts_nx2 + 0.5)],
            contour_idx=-1,
            color=1,
            thickness=-1,
            lineType=cv2.LINE_8,
        )
        area = int(np.sum(img))
        return area

    @property
    def perimeter(self) -> int:
        """Computes the perimeter"""
        if self._pts_nx2.shape[0] <= 2:
            perimeter = cv2.arcLength(self._pts_nx2.astype(np.float32), closed=False)
        else:
            perimeter = cv2.arcLength(self._pts_nx2.astype(np.float32), closed=True)
        return perimeter

    @property
    def roundness(self) -> float:
        """Compute roundness of a closed contour"""
        return roundness(self._pts_nx2)

    @property
    def is_convex(self) -> bool:
        """Tests a contour convexity."""
        return cv2.isContourConvex(self._pts_nx2)

    @property
    def key_points_nx2(self) -> np.ndarray:
        return self._pts_nx2

    @property
    def bounding_rect(self):
        x, y, w, h = cv2.boundingRect(np.float32(self._pts_nx2))
        return x, y, w, h

    @property
    def min_area_rect(self):
        """return param of smallest bounding rectangle(rotate)."""
        center_xy, rect_wh, angle_deg = cv2.minAreaRect(np.float32(self._pts_nx2))
        return center_xy, rect_wh, angle_deg

    @property
    def max_inner_circle(self):
        """return param of maximum inscribed circle."""
        # todo: 可能需要优化实现方法
        rect_x, rect_y, rect_w, rect_h = self.bounding_rect
        pts_nx2 = np.float32(self._pts_nx2)
        radius = 0
        center_xy = (0, 0)
        for x in range(rect_x, rect_x + rect_w):
            for y in range(rect_y, rect_y + rect_h):
                dist = cv2.pointPolygonTest(
                    contour=pts_nx2, pt=(x, y), measureDist=True
                )
                if dist >= radius:
                    radius = dist
                    center_xy = (x, y)
        return center_xy, radius

    @property
    def min_enclosing_circle(self):
        """return param of smallest outer circle."""
        center_xy, radius = cv2.minEnclosingCircle(np.float32(self._pts_nx2))
        return center_xy, radius

    @property
    def fit_ellipse(self):
        """
        return param of equivalent ellipse.
        Notes:
            PolygonROI points should be orderd, best used for external contours.
        """

        center_xy, axes_len, angle_deg = cv2.fitEllipse(np.float32(self._pts_nx2))
        return center_xy, axes_len, angle_deg

    def convex_hull(self, clockwise=True) -> "PolygonROI":
        """Finds the convex hull of a point set."""
        hull = cv2.convexHull(np.int0(self._pts_nx2 + 0.5), clockwise=clockwise)
        return PolygonROI(hull[:, 0, :])

    def copy(self) -> "PolygonROI":
        """Copy the polygon ROI"""
        return PolygonROI(self._pts_nx2)

    def draw(self, img, color, thickness, offset=(0, 0)):
        draw_contours(
            img,
            [np.int0(self._pts_nx2 + 0.5)],
            contour_idx=-1,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_8,
            offset=offset,
        )

    def skleton_map(self, map_wh):
        """return skeleton of area within contour"""
        img = np.zeros((map_wh[1], map_wh[0]), dtype=np.uint8)
        self.draw(
            img=img,
            color=(255,),
            thickness=-1,
            offset=(0, 0),
        )
        skleton = cv2.ximgproc.thinning(img)
        return skleton

    def transform(
        self,
        translation=(0, 0),
        rotate_angle_deg=0,
        rot_center=None,
        return_trans_M=False,
    ):
        if rot_center is None:
            rot_center = self.cxcy

        # Positive values mean counter-clockwise rotation
        matrix = cv2.getRotationMatrix2D(
            center=rot_center, angle=-rotate_angle_deg, scale=1
        )
        matrix[:, 2] += translation
        matrix = np.concatenate([matrix, np.array([[0.0, 0.0, 1.0]])], axis=0)

        self._pts_nx2 = project_pts(pts_dxn=self._pts_nx2.T, project_matrix=matrix).T

        if not return_trans_M:
            return None

        return matrix

    def scale(self, fx, fy, return_trans_M=False):
        """Scale(inplace) rectangle

        Args:
            fx (float): a positive number, scaling factor of width
            fy (float): a positive number, scaling factor of height
            return_trans_M (bool): whether to return the transform matrix

        """
        if fx < 0:
            raise ValueError(f"'fx' must be positive. fx: {fx}")

        if fy is None:
            fy = fx
        elif fy < 0:
            raise ValueError(f"'fy' must be positive. fy: {fy}")

        matrix = np.array([[fx, 0, 0], [0, fy, 0], [0, 0, 1]], dtype=np.float32)
        self._pts_nx2 = project_pts(self._pts_nx2.T, matrix).T

        if not return_trans_M:
            return None

        return matrix

    def expand(self, expand_percent=None, expand_pixel=None, return_trans_M=False):
        """
        Expand(inplace) the PolygonROI.

        Args:
            expand_percent: expand 'expand_percent' * dist(cxcy-->pt) pixels on each points
            expand_pixel: expand 'expand_pixel' pixels on each points
            return_trans_M (bool): whether to return the transform matrix

        Returns:
            matrix(np.ndarray): the transformation matrix

        Notes:
            Points whose expand pixel is lower than dist(cxcy-->pt)*-1 will shrink to cxcy.

        """
        if self._pts_nx2.shape[0] == 1:
            return np.diag([1.0, 1.0, 1.0])

        cx, cy = self.cxcy
        dist_from_cxcy = self._pts_nx2 - self.cxcy
        dist_from_cxcy = np.sqrt(np.power(dist_from_cxcy, 2).sum(1)).reshape(-1)

        if expand_pixel is not None:
            expand_percent = expand_pixel / dist_from_cxcy
        elif expand_percent is None:
            raise ValueError(
                "'expand_pixel' and 'expand_percent' could not be None at the same time."
            )

        expand_percent = np.maximum(np.array(expand_percent), -1)

        # expand
        org_pts = self._pts_nx2
        self._pts_nx2 = self._pts_nx2 - [cx, cy]
        self._pts_nx2 *= np.array([1 + expand_percent, 1 + expand_percent]).T
        self._pts_nx2 += [cx, cy]

        effective_loc = np.array(expand_percent) > -1

        if not return_trans_M:
            return None

        if np.sum(effective_loc) > 1:
            matrix, _ = cv2.findHomography(org_pts, self._pts_nx2)
        else:
            matrix = np.diag([1.0, 1.0, 1.0])

        return matrix

    def contain(self, xy):
        xy = tuple(np.array(xy, "float").reshape(2))

        # -1: out, 1: inner, 0: edge
        flag = cv2.pointPolygonTest(self._pts_nx2, xy, measureDist=False)

        return flag >= 0

    def dist(self, pts_nx2, measure_dist=True) -> List:
        check_array(shape=(-1, 2), pts_nx2=pts_nx2)
        pts = np.array(pts_nx2, "float")

        dist_list = []
        roi_pts_nx2 = self._pts_nx2

        for xy in pts:
            dist = cv2.pointPolygonTest(
                roi_pts_nx2.astype(np.float32), tuple(xy), measureDist=measure_dist
            )
            dist_list.append(dist)

        return dist_list

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

    def jitter(self, x_jitter, y_jitter=None, return_trans_M=False):
        """Jitter (inplace) each point in the PolygonROI

        Args:
            x_jitter: jitter for the points in x direction. Size of 'x_jitter'
                must equal to the number of points.
            y_jitter: jitter for the points in x direction. Size of 'y_jitter'
                must equal to the number of points.
            return_trans_M (bool): whether to return the transform matrix

        Returns:
            matrix(np.ndarray): the transformation matrix.

        Notes:
            Jitter changes some invariants of projective transformation,
            so the projective error calculated by 'cv2.findHomography'
            might be unsatisfied.

        """

        if y_jitter is None:
            y_jitter = x_jitter

        x_jitter = np.array(x_jitter).ravel()
        y_jitter = np.array(y_jitter).ravel()

        check_array(size=self._pts_nx2.shape[0], x_jitter=x_jitter, y_jitter=y_jitter)

        jitter = np.array([x_jitter, y_jitter]).T
        jitter_pts_nx2 = self._pts_nx2 + jitter
        self._pts_nx2 = jitter_pts_nx2

        if not return_trans_M:
            return None

        matrix, _ = cv2.findHomography(self._pts_nx2, jitter_pts_nx2)
        return matrix
