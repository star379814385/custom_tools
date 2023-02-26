import cv2
import numpy as np
from .cal_tools import cal_bbox_ious, cal_mask_ious
import pycocotools.mask as coco_mask


class DetEvaluator:
    def __init__(self):
        self.gts_list = []
        self.dts_list = []
        self.ious_list = []
        self.metric = None

    def add_data(self, gts_mx4, dts_nx4):
        assert isinstance(gts_mx4, np.ndarray) and isinstance(dts_nx4, np.ndarray)
        self.gts_list.append(gts_mx4)
        self.dts_list.append(dts_nx4)

    def evaluate(self):
        self.ious_list = [
            cal_bbox_ious(gts, dts) for gts, dts in zip(self.gts_list, self.dts_list)
        ]

    def get_metric(self, iou_thr):
        all_n_gts = 0
        all_n_dts = 0
        all_n_re = 0
        all_n_pr = 0
        tp_ious = []
        for ious in self.ious_list:
            n_gts, n_dts = ious.shape[:2]
            tp = ious >= iou_thr
            tp_ious.extend(list(ious[tp]))
            n_re = np.sum(np.min(tp, axis=1))
            n_pr = np.sum(np.min(tp, axis=0))
            all_n_gts += n_gts
            all_n_dts += n_dts
            all_n_re += n_re
            all_n_pr += n_pr
        mean_tp_iou = round(float(np.mean(tp_ious)), 2) if len(tp_ious) > 0 else 0
        recall = round(all_n_re / all_n_gts, 4) if all_n_gts > 0 else 0
        precision = round(all_n_pr / all_n_dts, 4) if all_n_dts > 0 else 0
        f1score = round((2 * recall * precision) / (recall + precision), 4) if recall + precision > 0 else 0
        metric = {
            "召回率": recall,
            "精准率": precision,
            "f1score": f1score,
            "正确检测平均iou": round(mean_tp_iou, 2),
            "正确召回数量": all_n_re,
            "精准预测数量": all_n_pr,
            "标注数量": all_n_gts,
            "预测数量": all_n_dts,
        }
        self.metric = metric
        return metric

    def reset(self):
        self.gts_list = []
        self.dts_list = []
        self.ious_list = []
        self.metric = None


class InsSegEvaluator(DetEvaluator):
    def evaluate(self):
        # self.ious_list = [
        #     cal_mask_ious(gts.reshape(gts.shape[0], -1), dts.reshape(dts.shape[0], -1))
        #     for gts, dts in zip(self.gts_list, self.dts_list)
        # ]
        def func_(gts, dts):
            gts = coco_mask.decode(gts)
            gts = gts.reshape((gts.shape[0], -1)).astype(np.bool_)
            dts = coco_mask.decode(dts)
            dts = dts.reshape((dts.shape[0], -1)).astype(np.bool_)
            return cal_mask_ious(gts, dts)
        self.ious_list = [func_(gts, dts) for gts, dts in zip(self.gts_list, self.dts_list)]

    def add_data(self, gts_mxhxw, dts_nxhxw):
        # self.gts_list.append(gts_mxhxw)
        # self.dts_list.append(dts_nxhxw)
        assert isinstance(gts_mxhxw, np.ndarray) and isinstance(dts_nxhxw, np.ndarray)
        self.gts_list.append(coco_mask.encode(gts_mxhxw))
        self.dts_list.append(coco_mask.encode(dts_nxhxw))

class KeyPointEvaluator:
    def __init__(self, points_num):
        self.points_num = points_num
        self.gts_list = []
        self.dts_list = []
        self.error_list = []
        self.metric = None

    def add_data(self, gt_mxpx2, dt_nxpx2):
        assert gt_mxpx2.ndim == dt_nxpx2.ndim == 3
        assert gt_mxpx2.shape[1] == dt_nxpx2.shape[1] == self.points_num
        self.gts_list.append(gt_mxpx2)
        self.dts_list.append(dt_nxpx2)

    def evaluate(self):
        for gt_mxpx2, dt_nxpx2 in zip(self.gts_list, self.dts_list):
            m = gt_mxpx2.shape[0]
            n = dt_nxpx2.shape[0]
            gt_mnxpx2 = np.repeat(gt_mxpx2[None], repeats=n, axis=0).reshape((m*n, p, 2))
            dt_mnxpx2 = np.repeat()



    def get_metric(self):
        pass

    def reset(self):
        self.gts_list = []
        self.dts_ilst = []
        self.metric = None






class ContourEvaluator:
    def __init__(self, cal_points_num=360):
        assert cal_points_num >= 3
        self.cal_points_num = cal_points_num
        self.gts_list = []
        self.dts_list = []
        self.gts_align_cx2_list = []
        self.dts_align_cx2_list = []
        self.max_error_per_ins = None
        self.mean_error_per_ins = None

    def add_data(self, gts_mx2_list, dts_nx2_list):
        for gts_mx2 in gts_mx2_list:
            assert gts_mx2.shape[0] >= 3
        for dts_nx2 in dts_nx2_list:
            assert dts_nx2.shape[0] >= 3
        self.gts_list.append(gts_mx2_list)
        self.dts_list.append(dts_nx2_list)

    def _get_align_points(self, points):
        M = cv2.moments(points)
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        points[:, 0] -= cx
        points[:, 1] -= cy
        magnitudes, angle_rads = cv2.cartToPolar(points[:, 0], points[:, 1])
        magnitudes = magnitudes.flatten()
        angle_rads = angle_rads.flatten()
        min_angle_rad_index = np.argmin(angle_rads)
        min_angle_rad = angle_rads[min_angle_rad_index]








class LineEvaluator:
    pass
