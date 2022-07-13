import numpy as np
from .cal_tools import cal_bbox_ious, cal_mask_ious


class DetEvaluator:
    def __init__(self):
        self.gts_list = []
        self.dts_list = []
        self.ious_list = []

    def add_data(self, gts_mx4, dts_nx4):
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
        mean_tp_iou = float(np.mean(tp_ious))
        eval_ = {
            "召回率": round(all_n_re / all_n_gts, 2),
            "精准率": round(all_n_pr / all_n_dts, 2),
            "正确检测平均iou": round(mean_tp_iou, 2),
            "正确召回数量": all_n_re,
            "精准预测数量": all_n_pr,
            "标注数量": all_n_gts,
            "预测数量": all_n_dts,
        }
        return eval_

    def reset(self):
        self.gts_list = []
        self.dts_list = []
        self.ious_list = []


class InsSegEvaluator(DetEvaluator):
    def evaluate(self):
        self.ious_list = [
            cal_mask_ious(gts.view(gts.shape[0], -1), dts.view(dts.shape[0], -1))
            for gts, dts in zip(self.gts_list, self.dts_list)
        ]


class ContourEvaluator:
    pass


class LineEvaluator:
    pass
