import cv2
import torch
import numpy as np
from deploy.single_stage_detector import SingleStageDetectorLibtorch
import torch.nn.functional as F
from typing import List
import math

def torch_cal_bbox_ious(bbox_mx4: torch.Tensor, bbox_nx4: torch.Tensor):
    m, n = bbox_mx4.shape[0], bbox_nx4.shape[0]
    gts_mnx4 = bbox_mx4.repeat(n, 1).reshape(m * n, -1)
    dts_mnx4 = bbox_nx4[None].repeat(m, 1, 1).reshape(m * n, -1)

    x1_max = torch.max(torch.stack((gts_mnx4[:, 0], dts_mnx4[:, 0]), dim=0), dim=0)[0]
    y1_max = torch.max(torch.stack((gts_mnx4[:, 1], dts_mnx4[:, 1]), dim=0), dim=0)[0]
    x2_min = torch.min(torch.stack((gts_mnx4[:, 2], dts_mnx4[:, 2]), dim=0), dim=0)[0]
    y2_min = torch.min(torch.stack((gts_mnx4[:, 3], dts_mnx4[:, 3]), dim=0), dim=0)[0]

    and_ = torch.clip(x2_min - x1_max, min=0, max=math.inf) * torch.clip(y2_min - y1_max, min=0, max=math.inf)
    or_ = (gts_mnx4[:, 2] - gts_mnx4[:, 0]) * (gts_mnx4[:, 3] - gts_mnx4[:, 1]) + (
        dts_mnx4[:, 2] - dts_mnx4[:, 0]
    ) * (dts_mnx4[:, 3] - dts_mnx4[:, 1]) - and_

    ious = (and_ / (or_ + 1e-8)).reshape((m, n))
    return ious


def nms(boxes, scores, idxs, iou_thr: float = 0.5, max_num: int = 100, ignore_cls: bool = False):
    sort_ids = torch.argsort(scores, descending=True)[:max_num]
    boxes = boxes[sort_ids]
    scores = scores[sort_ids]
    idxs = idxs[sort_ids]

    dst_boxes = []
    dst_scores = []
    dst_idxs = []
    for cls_id in torch.unique(idxs):
        if ignore_cls:
            keeps = idxs >= 0
        else:
            keeps = idxs == cls_id
        cls_boxes = boxes[keeps]
        cls_scores = scores[keeps]
        cls_idxs = idxs[keeps]
        while cls_boxes.numel() > 0:
            dst_scores.append(cls_scores[0])
            dst_boxes.append(cls_boxes[0])
            dst_idxs.append(cls_idxs[0])
            cls_boxes = cls_boxes[1:]
            cls_scores = cls_scores[1:]
            cls_idxs = cls_idxs[1:]
            if cls_boxes.numel() == 0:
                break
            ious = torch_cal_bbox_ious(bbox_mx4=dst_boxes[-1][None], bbox_nx4=cls_boxes)[0]
            keeps = ious < iou_thr
            cls_boxes = cls_boxes[keeps]
            cls_scores = cls_scores[keeps]
            cls_idxs = cls_idxs[keeps]
        if ignore_cls:
            break

    dst_boxes = torch.stack(dst_boxes, dim=0)
    dst_scores = torch.stack(dst_scores)
    dst_boxes = torch.cat([dst_boxes, dst_scores[:, None]], dim=-1)
    dst_idxs = torch.stack(dst_idxs)
    return dst_boxes, dst_idxs


class YOLOXLibtorch(SingleStageDetectorLibtorch):

    @classmethod
    def build_from_mmdet_module(
        cls, model, inputs_bchw, device="cuda:0"
    ):
        model.to(device).eval()
        x = inputs_bchw.to(device)
        # backbone
        backbone = torch.jit.trace(model.backbone, x)
        with torch.no_grad():
            feats = backbone(x)

        # neck
        if model.with_neck:
            assert model.neck.__class__.__name__ == "YOLOXPAFPN"
            neck = torch.jit.trace(model.neck, (feats,))
            with torch.no_grad():
                feats = neck(feats)
        else:
            neck = None

        # bbox_head
        assert model.bbox_head.__class__.__name__ == "YOLOXHead"
        bbox_head_forward = torch.jit.trace(model.bbox_head, (feats,))
        modified_bbox_head = YOLOXHeadLibtorch(
            bbox_head_forward=bbox_head_forward,
            prior_generator=model.bbox_head.prior_generator,
            cls_out_channels=model.bbox_head.cls_out_channels,
            score_thr=0.05,
            iou_thr=0.65,
            max_num=10000,
        )
        # with torch.no_grad():
        #     outs = modified_bbox_head(feats)
        #     print(outs)
        bbox_head = torch.jit.trace(modified_bbox_head, (feats, ))
        with torch.no_grad():
            outs = bbox_head(feats)
            print(outs)
        # exit()
        return cls(backbone, neck, bbox_head)



class YOLOXHeadLibtorch(torch.nn.Module):
    def __init__(self, bbox_head_forward, prior_generator, cls_out_channels, score_thr=0.05, iou_thr=0.65, max_num=100,):
        super(YOLOXHeadLibtorch, self).__init__()
        self.bbox_head_forward = bbox_head_forward
        self.prior_generator = prior_generator
        self.cls_out_channels = cls_out_channels
        self.score_thr = score_thr
        self.iou_thr = iou_thr
        self.max_num = max_num

    def _bbox_decode(self, priors, bbox_preds):
        xys = (bbox_preds[..., :2] * priors[:, 2:]) + priors[:, :2]
        whs = bbox_preds[..., 2:].exp() * priors[:, 2:]

        tl_x = (xys[..., 0] - whs[..., 0] / 2)
        tl_y = (xys[..., 1] - whs[..., 1] / 2)
        br_x = (xys[..., 0] + whs[..., 0] / 2)
        br_y = (xys[..., 1] + whs[..., 1] / 2)

        decoded_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)
        return decoded_bboxes


    def forward(self, feats):
        # 只支持单图推断，不支持rescale
        cls_scores, bbox_preds, objectnesses = self.bbox_head_forward(feats)
        flatten_cls_scores, flatten_objectness, flatten_bboxes = self.bbox_decode(cls_scores, bbox_preds, objectnesses)
        results = self.get_bboxes(flatten_cls_scores, flatten_objectness, flatten_bboxes, self.score_thr, self.iou_thr, self.max_num)

        return results

    @staticmethod
    @torch.jit._script_if_tracing
    def get_bboxes(flatten_cls_scores, flatten_objectness, flatten_bboxes, score_thr: float, iou_thr: float, max_num: int):
        # 每个实例的编码为：x1, y1, x2, y2, scores, id
        results = []
        for img_id in range(flatten_cls_scores.shape[0]):
            cls_scores = flatten_cls_scores[img_id]
            score_factor = flatten_objectness[img_id]
            bboxes = flatten_bboxes[img_id]
            dets, labels = YOLOXHeadLibtorch._bboxes_nms(
                    cls_scores, bboxes, score_factor, score_thr=score_thr, iou_thr=iou_thr, max_num=max_num
                )
            if dets.numel() == 0:
                result = torch.zeros((0, 6), dtype=torch.float32)
            else:
                result = torch.cat((dets, labels[:, None]), dim=-1)
            results.append(result)
        return results


    @staticmethod
    @torch.jit._script_if_tracing
    def _bboxes_nms(cls_scores, bboxes, score_factor, score_thr: float=0.05, iou_thr: float=0.65, max_num: int=10000):
        max_scores, labels = torch.max(cls_scores, 1)
        valid_mask = score_factor * max_scores >= score_thr

        bboxes = bboxes[valid_mask]
        scores = max_scores[valid_mask] * score_factor[valid_mask]
        labels = labels[valid_mask]

        if labels.numel() == 0:
            return bboxes, labels
        else:
            # dets, keep = batched_nms(bboxes, scores, labels, cfg.nms)
            # return dets, labels[keep]
            # 目前固定使用普通的nms
            dets, labels = nms(bboxes, scores, labels, iou_thr=iou_thr, max_num=max_num, ignore_cls=False)
            return dets, labels

    @staticmethod
    @torch.jit._script_if_tracing
    def get_featmap_sizes(cls_scores: List[torch.Tensor]):
        # return [cls_score.shape[2:] for cls_score in cls_scores]
        return [torch.tensor(cls_score.shape[2:]) for cls_score in cls_scores]


    def bbox_decode(self,
                   cls_scores,
                   bbox_preds,
                   objectnesses,
                   ):
        assert len(cls_scores) == len(bbox_preds) == len(objectnesses)

        # assert cls_scores[0].shape[0] == 1 # 只支持单图推断
        featmap_sizes = self.get_featmap_sizes(cls_scores)
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(1, -1,
                                                  self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(1, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(1, -1)
            for objectness in objectnesses
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        flatten_priors = torch.cat(mlvl_priors)

        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)

        return flatten_cls_scores, flatten_objectness, flatten_bboxes


def preprocess(data):
    img = data["img"]
    input_hw = data["input_hw"]
    ori_h, ori_w = img.shape[:2]
    # resize(keep ratio)
    scale_h = input_hw[0] / ori_h
    scale_w = input_hw[1] / ori_w
    scale = min(scale_w, scale_h)
    cur_h, cur_w = int(ori_h * scale), int(ori_w * scale)
    img = cv2.resize(img, (cur_w, cur_h))
    # padding
    img = cv2.copyMakeBorder(
        img, 0, input_hw[0] - cur_h, 0, input_hw[1] - cur_w,
        borderType=cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    # show_image(img)
    # totensor
    img_tensor = torch.from_numpy(img).to(torch.float32).permute(2, 0, 1)[None]

    data["scale"] = scale
    data["img_tensor"] = img_tensor

    return data

if __name__ == "__main__":
    from mmdet.apis.inference import init_detector, inference_detector
    from custom_tools.cvutils import read_image, show_image
    # from deploy._test_utils import preprocess

    cfg_path = r"D:\project\2_tianhua_lvmo\model\20220906_yolox_s_det_cls2_so2\yolox_s_960_cls2_so2.py"
    checkpoint_path = r"D:\project\2_tianhua_lvmo\model\20220906_yolox_s_det_cls2_so2\yolox_s_960_cls2_so2_save_dir\best_0.5_F1_epoch_300.pth"


    save_jit_path = "yolox_torch181_cpu.pt"
    img_path = r"D:\project\2_tianhua_lvmo\dataset\images\longgu_origin_20220701\CeliangRobot-6D20F-01-1-002.png"
    device = "cpu"
    # device = "cuda"
    input_shape_hw = (960, 960)
    batch_size = 1

    img = read_image(img_path)
    model = init_detector(cfg_path, checkpoint=checkpoint_path)
    # results = inference_detector(model, [img])
    data = dict(
        img=img,
        input_hw=input_shape_hw,
    )
    data = preprocess(data)

    imgs_tensor = data["img_tensor"]

    modified_yolox = YOLOXLibtorch.build_from_mmdet_module(
        model=model,
        inputs_bchw=imgs_tensor,
        device=device,
    )
    # delattr(modified_yolox, "build_from_mmdet_module")
    yolox_scripts = torch.jit.script(modified_yolox)
    torch.jit.save(yolox_scripts, save_jit_path)
    exit()

    yolox_scripts = torch.jit.load(save_jit_path)
    for name, modules in yolox_scripts.named_modules():
        print(name)
    yolox_scripts.named_modules()
    imgs_tensor = imgs_tensor.cuda()
    yolox_scripts.cuda()
    results = yolox_scripts(imgs_tensor)
    for result in results:
        result[:, :4] /= data["scale"]
        for x1, y1, x2, y2, score, cate_id in result:
            if cate_id == 0:
                continue
            if score < 0.4:
                continue
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=3)
        show_image(img)


