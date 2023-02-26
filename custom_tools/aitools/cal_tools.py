import numpy as np

__all__ = [
    "cal_bbox_ious",
    "cal_mask_ious",
]


def cal_bbox_ious(bbox_mx4, bbox_nx4):
    m, n = bbox_mx4.shape[0], bbox_nx4.shape[0]
    gts_mnx4 = np.repeat(bbox_mx4, n, 0).reshape(m * n, -1)
    dts_mnx4 = np.repeat(bbox_nx4[None], m, 0).reshape(m * n, -1)
    x1_min, x1_max, y1_min, y1_max, x2_min, x2_max, y2_min, y2_max = [
        func((gts_mnx4[:, i], dts_mnx4[:, i]), axis=0)
        for i in range(4)
        for func in (np.min, np.max)
    ]
    and_ = np.clip(x2_min - x1_max, a_min=0, a_max=np.inf) * np.clip(y2_min - y1_max, a_min=0, a_max=np.inf)
    or_ = (gts_mnx4[:, 2] - gts_mnx4[:, 0]) * (gts_mnx4[:, 3] - gts_mnx4[:, 1]) + (
        dts_mnx4[:, 2] - dts_mnx4[:, 0]
    ) * (dts_mnx4[:, 3] - dts_mnx4[:, 1]) - and_

    ious = (and_ / (or_ + 1e-8)).reshape((m, n))
    return ious

def cal_mask_ious(mask_mxhw, mask_nxhw):
    m, n = mask_mxhw.shape[0], mask_nxhw.shape[0]
    mask_mxhw = mask_mxhw.astype(np.bool_)
    mask_nxhw = mask_nxhw.astype(np.bool_)
    mask_mnxhw1 = np.repeat(mask_mxhw, n, 0).reshape(m * n, -1)
    mask_mnxhw2 = np.repeat(mask_nxhw[None], m, 0).reshape(m * n, -1)

    and_ = np.sum(np.bitwise_and(mask_mnxhw1, mask_mnxhw2), axis=-1)
    or_ = np.sum(np.bitwise_or(mask_mnxhw1, mask_mnxhw2), axis=-1)

    ious = (and_ / (or_ + 1e-8)).reshape((m, n))
    return ious

