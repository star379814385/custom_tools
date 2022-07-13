import copy
import math
from typing import Tuple, Union

import cv2
import numpy as np

__all__ = [
    "BaseFrequencyModifier",
    "WaveletGenerator",
]


def normalize(data: np.ndarray, dtype=np.uint8):
    min_ = np.min(data)
    max_ = np.max(data)
    data_normalize = ((data - min_) * 255 / (max_ - min_)).astype(dtype)
    return data_normalize


# 1.gray mapping
class GrayMappingBaseDefectSynthesizer(object):
    def __init__(
        self,
        src_grayvalue_range: Tuple[float, int],
        dst_threshold_range: Tuple[float, int],
        mask_threshold: int,
    ):
        _k = (dst_threshold_range[1] - dst_threshold_range[0]) / (
            src_grayvalue_range[1] - src_grayvalue_range[0]
        )
        _b = dst_threshold_range[0] - src_grayvalue_range[0] * _k

        def _f(src: np.ndarray) -> np.ndarray:
            dst = (src.astype(np.float32) * _k + _b).clip(
                dst_threshold_range[0], dst_threshold_range[1]
            )
            # dst = np.expand_dims(dst, -1).repeat(3, -1)
            return dst

        self._graymapping = _f
        self._mask_threshold = mask_threshold

    @staticmethod
    def _op(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _gen_mask(src: np.ndarray, dst: np.ndarray, threshold: int):
        dif = np.abs(src.astype(np.float32) - dst.astype(np.float32)) > threshold
        return dif.astype(np.uint8) * 255

    def run(
        self,
        image: np.ndarray,
        image_texture: np.ndarray,
    ):
        assert image.ndim == 3 and image_texture.ndim == 3
        h, w = image.shape[:2]
        image_texture = cv2.resize(
            image_texture, (w, h), interpolation=cv2.INTER_LINEAR
        )
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image_gray = image_hsv[..., -1]
        image_texture = cv2.cvtColor(image_texture, cv2.COLOR_BGR2GRAY)

        image_texture_dst = self._graymapping(image_texture)
        image_gray_synthesize = self._op(image_gray, image_texture_dst)
        mask = self._gen_mask(image_gray, image_gray_synthesize, self._mask_threshold)

        image_hsv_dst = image_hsv
        image_hsv_dst[..., -1] = image_gray_synthesize
        image_synthesize = cv2.cvtColor(image_hsv_dst, cv2.COLOR_HSV2BGR)
        if mask.ndim == 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return image_synthesize, mask


class GrayMappingAddDefectSynthesizer(GrayMappingBaseDefectSynthesizer):
    @staticmethod
    def _op(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        return (
            (image1.astype(np.float32) + image2.astype(np.float32))
            .clip(0, 255)
            .astype(np.uint8)
        )


class GrayMappingMulDefectSynthesizer(GrayMappingBaseDefectSynthesizer):
    @staticmethod
    def _op(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        return (
            (image1.astype(np.float32) * image2.astype(np.float32))
            .clip(0, 255)
            .astype(np.uint8)
        )


# 2. frequency modify
class BaseFrequencyModifier(object):
    def __init__(
        self,
    ):
        self.magnitude_input = None

    def _op(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def run(
        self,
        image: np.ndarray,
        *args,
        **kwargs,
    ):
        # if image.ndim == 3:
        #     image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #     image_gray = image_hsv[..., -1]
        # else:
        #     image_gray = image

        image_gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        )

        # FFT
        dft_shift_src = np.fft.fftshift(np.fft.fft2(image_gray))
        magnitude_src = np.abs(dft_shift_src)
        _, phase_src = cv2.cartToPolar(dft_shift_src.real, dft_shift_src.imag)
        magnitude_log_src = np.log(magnitude_src)

        # op
        self.magnitude_input = copy.deepcopy(magnitude_log_src)
        magnitude_output = self._op(*args, **kwargs)

        # to spa
        magnitude_dst = np.exp(magnitude_output)

        dft_shift_dst = dft_shift_src
        dft_shift_dst.real, dft_shift_dst.imag = cv2.polarToCart(
            magnitude_dst, phase_src
        )

        image_gray_dst = np.fft.ifft2(np.fft.ifftshift(dft_shift_dst))
        image_gray_dst = image_gray_dst.real.clip(0, 255).astype(np.uint8)

        # if image.ndim == 3:
        #     image_hsv[..., -1] = image_gray_dst
        #     image_dst = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
        # else:
        #     image_dst = image_gray_dst
        image_dst = (
            np.dstack([image_gray_dst, image_gray_dst, image_gray_dst])
            if image.ndim == 3
            else image_gray_dst
        )

        return image_dst


class WaveletGenerator(BaseFrequencyModifier):
    def __init__(
        self,
        ksize_wh: Tuple,
        dist_range: Union[Tuple, float],
        angle_deg_range: Union[Tuple, float],
    ):
        self.ksize_wh = ksize_wh
        self.dist_range = dist_range
        self.angle_deg_range = angle_deg_range

    def _op(  # type: ignore
        self,
        mode=0,
        params=[
            1.4,
        ],
    ) -> np.ndarray:
        magnitude_output = self.magnitude_input
        if isinstance(self.angle_deg_range, tuple):
            angle_deg = min(self.angle_deg_range) + np.random.random() * (
                max(self.angle_deg_range) - min(self.angle_deg_range)
            )
        else:
            angle_deg = self.angle_deg_range
        if isinstance(self.dist_range, tuple):
            dist_pixel = min(self.dist_range) + np.random.random() * (
                max(self.dist_range) - min(self.dist_range)
            )
        else:
            dist_pixel = self.dist_range

        coord_xy_wrt_center = (
            round(dist_pixel * math.cos(angle_deg)),
            round(dist_pixel * math.sin(angle_deg)),
        )

        magnitude_output_center_xy = (
            magnitude_output.shape[1] // 2,
            magnitude_output.shape[0] // 2,
        )
        # gen noise in spectrum_map
        noise1_x1, noise1_y1 = (
            magnitude_output_center_xy[0]
            - coord_xy_wrt_center[0]
            - self.ksize_wh[0] // 2,
            magnitude_output_center_xy[1]
            - coord_xy_wrt_center[1]
            - self.ksize_wh[1] // 2,
        )
        noise2_x2, noise2_y2 = (
            magnitude_output_center_xy[0]
            + coord_xy_wrt_center[0]
            + self.ksize_wh[0] // 2,
            magnitude_output_center_xy[1]
            + coord_xy_wrt_center[1]
            + self.ksize_wh[1] // 2,
        )

        # mode 1: todo
        # kernel = np.random.normal(np.max(magnitude_output) * 0.1, 30,
        # ksize_xy[
        # ::-1])
        # kernel = magnitude_output[noise1_y1: noise1_y1 + ksize_xy[1],
        # noise1_x1:
        # noise1_x1
        # + ksize_xy[0]] * 1.5
        if mode == 0:
            kernel = (
                magnitude_output[
                    noise1_y1 : noise1_y1 + self.ksize_wh[1],
                    noise1_x1 : noise1_x1 + self.ksize_wh[0],
                ]
                * params[0]
            )
            kernel = kernel.clip(np.min(magnitude_output), np.max(magnitude_output))

        magnitude_output[
            noise1_y1 : noise1_y1 + self.ksize_wh[1],
            noise1_x1 : noise1_x1 + self.ksize_wh[0],
        ] = kernel
        magnitude_output[
            noise2_y2 - self.ksize_wh[1] : noise2_y2,
            noise2_x2 - self.ksize_wh[0] : noise2_x2,
        ] = kernel[::-1, ::-1]

        return magnitude_output


#
# class ChaoPiDefectSynthesizer(object):
#     def __init__(
#         self,
#         ksize_wh: Tuple,
#         dist_range: Tuple,
#         angle_deg_range: Tuple,
#         mask_threshold: int,
#     ):
#         self.frequency_modifier = ChaoPiFrequencyModifier(
#             ksize_wh, dist_range, angle_deg_range
#         )
#         self._mask_threshold = mask_threshold
#
#     def run(self, image: np.ndarray):
#         assert image.ndim == 3
#         image_src = image
#         image_dst = self.frequency_modifier.run(image_src)
#
#         image_gray_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
#         image_gray_dst = cv2.cvtColor(image_dst, cv2.COLOR_BGR2GRAY)
#
#         mask = self._gen_mask(image_gray_src, image_gray_dst, self._mask_threshold)
#         if mask.ndim == 2:
#             mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
#         return image_dst, mask
#
#     @staticmethod
#     def _gen_mask(src: np.ndarray, dst: np.ndarray, threshold: int):
#         dif = np.abs(src.astype(np.float32) - dst.astype(np.float32)) > threshold
#         return dif.astype(np.uint8) * 255


# # 3.
# class ScratchDefectSynthesizer:
#     def __init__(
#         self,
#         beta_range=(50, 70),
#         thickness_range=(10, 20),  # 线粗细
#         length_range=(50, 200),  # 线长短
#         iter_range: Tuple[int, int] = (1, 1),
#     ):
#         self.synthesizer = BrightnessLineDefectSynthesizer(
#             beta_range=beta_range,
#             thickness_range=thickness_range,  # 线粗细
#             length_range=length_range,  # 线长短
#         )
#         self.iter_range = iter_range
#
#     def run(
#         self,
#         image: np.ndarray,
#     ):
#         if self.iter_range[0] == self.iter_range[1]:
#             iter_num = self.iter_range[0]
#         else:
#             iter_num = iter_num = np.random.randint(*self.iter_range)
#         image_synthesize = copy.deepcopy(image)
#         # mask = np.zeros(image_synthesizer.shape[:2], dtype=np.uint8)
#         mask = np.zeros_like(image_synthesize, dtype=np.uint8)
#         for _ in range(iter_num):
#             image_synthesize_temp, mask_temp = self.synthesizer.run(image)
#             image_synthesize = image_synthesize * (mask_temp == 0).astype(
#                 np.uint8
#             ) + image_synthesize_temp * (mask_temp > 0).astype(np.uint8)
#             mask = np.max([mask, mask_temp], axis=0)
#         return image_synthesize, mask
#
#
# class StainDefectSynthesizer:
#     def __init__(
#         self,
#         beta_range=(50, 70),  # beta变化的区间，自适应无效
#         size_range=(20, 40),  # 点大小的区间
#         iter_range: Tuple[int, int] = (1, 1),
#     ):
#         self.synthesizer = BrightnessDotDefectSynthesizer(
#             beta_range=beta_range,
#             size_range=size_range,
#         )
#         self.iter_range = iter_range
#
#     def run(self, image: np.ndarray):
#         if self.iter_range[0] == self.iter_range[1]:
#             iter_num = self.iter_range[0]
#         else:
#             iter_num = iter_num = np.random.randint(*self.iter_range)
#         image_synthesize = copy.deepcopy(image)
#         # mask = np.zeros(image_synthesizer.shape[:2], dtype=np.uint8)
#         mask = np.zeros_like(image_synthesize, dtype=np.uint8)
#         for _ in range(iter_num):
#             image_synthesize_temp, mask_temp = self.synthesizer.run(image)
#             image_synthesize = image_synthesize * (mask_temp == 0).astype(
#                 np.uint8
#             ) + image_synthesize_temp * (mask_temp > 0).astype(np.uint8)
#             mask = np.max([mask, mask_temp], axis=0)
#         return image_synthesize, mask
