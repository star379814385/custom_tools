import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np


def mask_matrix_nms(
    masks,
    labels,
    scores,
    filter_thr: float = -1,
    nms_pre: int = -1,
    max_num: int = -1,
    kernel: str = "gaussian",
    sigma: float = 2.0,
    mask_area: Optional[torch.Tensor] = None,
):
    # 更改nms部分代码并增加参数类型说明，使之符合torch.jit.scripts的语法
    """Matrix NMS for multi-class masks.

    Args:
        masks (Tensor): Has shape (num_instances, h, w)
        labels (Tensor): Labels of corresponding masks,
            has shape (num_instances,).
        scores (Tensor): Mask scores of corresponding masks,
            has shape (num_instances).
        filter_thr (float): Score threshold to filter the masks
            after matrix nms. Default: -1, which means do not
            use filter_thr.
        nms_pre (int): The max number of instances to do the matrix nms.
            Default: -1, which means do not use nms_pre.
        max_num (int, optional): If there are more than max_num masks after
            matrix, only top max_num will be kept. Default: -1, which means
            do not use max_num.
        kernel (str): 'linear' or 'gaussian'.
        sigma (float): std in gaussian method.
        mask_area (Tensor): The sum of seg_masks.

    Returns:
        tuple(Tensor): Processed mask results.

            - scores (Tensor): Updated scores, has shape (n,).
            - labels (Tensor): Remained labels, has shape (n,).
            - masks (Tensor): Remained masks, has shape (n, w, h).
            - keep_inds (Tensor): The indices number of
                the remaining mask in the input mask, has shape (n,).
    """
    assert len(labels) == len(masks) == len(scores)
    # if len(labels) == 0:
    #     return scores.new_zeros(0), labels.new_zeros(0), masks.new_zeros(
    #         0, *masks.shape[-2:]), labels.new_zeros(0)
    if len(labels) == 0:
        return (
            scores.new_zeros(0),
            labels.new_zeros(0),
            masks.new_zeros(0, masks.shape[-2], masks.shape[-1]),
            labels.new_zeros(0),
        )
    if mask_area is None:
        mask_area = masks.sum((1, 2)).float()
    else:
        assert len(masks) == len(mask_area)

    # sort and keep top nms_pre
    scores, sort_inds = torch.sort(scores, descending=True)

    keep_inds = sort_inds
    if nms_pre > 0 and len(sort_inds) > nms_pre:
        sort_inds = sort_inds[:nms_pre]
        keep_inds = keep_inds[:nms_pre]
        scores = scores[:nms_pre]
    masks = masks[sort_inds]
    mask_area = mask_area[sort_inds]
    labels = labels[sort_inds]

    num_masks = len(labels)
    flatten_masks = masks.reshape(num_masks, -1).float()
    # inter.
    inter_matrix = torch.mm(flatten_masks, flatten_masks.transpose(1, 0))
    expanded_mask_area = mask_area.expand(num_masks, num_masks)
    # Upper triangle iou matrix.
    iou_matrix = (
        inter_matrix
        / (expanded_mask_area + expanded_mask_area.transpose(1, 0) - inter_matrix)
    ).triu(diagonal=1)
    # label_specific matrix.
    expanded_labels = labels.expand(num_masks, num_masks)
    # Upper triangle label matrix.
    label_matrix = (expanded_labels == expanded_labels.transpose(1, 0)).triu(diagonal=1)

    # IoU compensation
    compensate_iou, _ = (iou_matrix * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(num_masks, num_masks).transpose(1, 0)

    # IoU decay
    decay_iou = iou_matrix * label_matrix

    # Calculate the decay_coefficient
    if kernel == "gaussian":
        decay_matrix = torch.exp(-1 * sigma * (decay_iou**2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou**2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == "linear":
        decay_matrix = (1 - decay_iou) / (1 - compensate_iou)
        decay_coefficient, _ = decay_matrix.min(0)
    else:
        raise NotImplementedError(f"{kernel} kernel is not supported in matrix nms!")

    # update the score.
    scores = scores * decay_coefficient

    if filter_thr > 0:
        keep = scores >= filter_thr
        keep_inds = keep_inds[keep]
        if not keep.any():
            # return scores.new_zeros(0), labels.new_zeros(0), masks.new_zeros(
            #     0, *masks.shape[-2:]), labels.new_zeros(0)
            return (
                scores.new_zeros(0),
                labels.new_zeros(0),
                masks.new_zeros(0, masks.shape[-2], masks.shape[-1]),
                labels.new_zeros(0),
            )
        masks = masks[keep]
        scores = scores[keep]
        labels = labels[keep]

    # sort and keep top max_num
    scores, sort_inds = torch.sort(scores, descending=True)
    keep_inds = keep_inds[sort_inds]
    if max_num > 0 and len(sort_inds) > max_num:
        sort_inds = sort_inds[:max_num]
        keep_inds = keep_inds[:max_num]
        scores = scores[:max_num]
    masks = masks[sort_inds]
    labels = labels[sort_inds]

    return scores, labels, masks, keep_inds


class ModuleWrapper(torch.nn.Module):
    """
    nn.Module输入存在tuple类型参数的module，无法被jit.trace正确追踪
    因此为这些Module提供一个用于转换的Module
    """

    def __init__(self, module):
        super(ModuleWrapper, self).__init__()
        self.module = module

    def forward(self, *args):
        x = [i for i in args]

        return self.module(x)


class SimpleSOLOV2(torch.nn.Module):
    # 由于torchscripts转换需要，将decoder(get_results)拆分为数个模块
    def __init__(
        self,
        backbone,
        neck,
        mask_head,
        kernel_preds_decoder,
        cls_preds_decoder,
        get_results_decoder,
    ):
        super(SimpleSOLOV2, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.mask_head = mask_head
        self.kernel_preds_decoder = kernel_preds_decoder
        self.cls_preds_decoder = cls_preds_decoder
        self.get_results_decoder = get_results_decoder

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.neck is not None:
            x = self.neck(*x)
        return x

    def forward(self, img):
        """
        Args:
            img (torch.Tensor): input data, with shape (b, 3, h, w)

        Returns:
            # results (List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]): with length batchsize,
            every element includes:
                results_scores (torch.float): with shape (n, )
                results_masks (torch.bool): with shape (n, h, w)
                results_labels (torch.int): with shape(n, )

        """
        img_h, img_w = int(img.shape[2]), int(img.shape[3])
        feats = self.extract_feat(img)
        outs = self.mask_head(*feats)

        # decode
        mlvl_kernel_preds, mlvl_cls_preds, mask_feats = outs[0], outs[1], outs[2]
        # assert len(mlvl_kernel_preds) == len(mlvl_cls_preds) == 5
        # kernel_preds = self.kernel_preds_decoder(*mlvl_kernel_preds)

        # todo: 不明白为什么必须显性写出各个参数，而mask_head不需要
        kernel_preds = self.kernel_preds_decoder(
            mlvl_kernel_preds[0],
            mlvl_kernel_preds[1],
            mlvl_kernel_preds[2],
            mlvl_kernel_preds[3],
            mlvl_kernel_preds[4],
            # *mlvl_kernel_preds
        )

        cls_preds = self.cls_preds_decoder(
            mlvl_cls_preds[0],
            mlvl_cls_preds[1],
            mlvl_cls_preds[2],
            mlvl_cls_preds[3],
            mlvl_cls_preds[4],
            # *mlvl_cls_preds,
        )


        results_list = self.get_results_decoder(
            kernel_preds, cls_preds, mask_feats, img_h, img_w
        )
        return results_list

    @classmethod
    def build_from_mmdet_module(
        cls, model, input_shape_hw, use_postprocess=True, device="cuda:0"
    ):
        model.to(device).eval()
        h, w = input_shape_hw
        x = torch.rand((1, 3, h, w), dtype=torch.float32).to(device)

        # backbone
        backbone = torch.jit.trace(model.backbone, x, strict=False)
        feats = backbone(x)

        # neck
        if model.with_neck:
            assert model.neck.__class__.__name__ in ["FPN"]
            neck_wrapper = ModuleWrapper(model.neck)
            neck = torch.jit.trace(neck_wrapper, feats, strict=False)
            feats = neck(*feats)
        else:
            neck = None

        # mask_head
        mask_head_wrapper = ModuleWrapper(model.mask_head)
        mask_head = torch.jit.trace(mask_head_wrapper, feats, strict=False)
        outs = mask_head(*feats)

        # decoder
        # get kernel decoder
        mlvl_kernel_preds = outs[0]
        kernel_preds_decoder = torch.jit.trace(
            ModuleWrapper(
                SOLOV2KernelPredsDecoder(
                    kernel_out_channels=model.mask_head.kernel_out_channels
                )
            ),
            mlvl_kernel_preds,
            strict=False,
        )
        kernel_preds = kernel_preds_decoder(*mlvl_kernel_preds)

        # get cls decoder
        mlvl_cls_preds = outs[1]
        cls_preds_decoder = torch.jit.trace(
            ModuleWrapper(
                SOLOV2ClsPredsDecoder(num_classes=model.mask_head.num_classes)
            ),
            mlvl_cls_preds,
            strict=False,
        )
        cls_preds = cls_preds_decoder(*mlvl_cls_preds)

        # get results
        mask_feats = outs[2]
        decoder = SimpleSOLOV2Decoder if use_postprocess else SimpleSOLOV2DecoderNoPost
        results_decoder = torch.jit.script(
            decoder(
                num_classes=model.mask_head.num_classes,
                mask_stride=model.mask_head.mask_stride,
                num_grids=model.mask_head.num_grids,
                strides=model.mask_head.strides,
                num_levels=model.mask_head.num_levels,
                dynamic_conv_size=model.mask_head.dynamic_conv_size,
                test_cfg=model.mask_head.test_cfg,
            ).to(device),
        )
        results = results_decoder(kernel_preds, cls_preds, mask_feats, h, w)

        return cls(
            backbone,
            neck,
            mask_head,
            kernel_preds_decoder,
            cls_preds_decoder,
            results_decoder,
        )


class SOLOV2KernelPredsDecoder(torch.nn.Module):
    def __init__(self, kernel_out_channels):
        super(SOLOV2KernelPredsDecoder, self).__init__()
        self.kernel_out_channels = kernel_out_channels

    def _get_img_kernel_preds(self, mlvl_kernel_preds):
        num_imgs = mlvl_kernel_preds[0].shape[0]
        num_levels = len(mlvl_kernel_preds)
        img_kernel_pred_list = []
        for img_id in range(num_imgs):
            img_kernel_pred = [
                # mlvl_kernel_preds[lvl][img_id]
                # .permute(1, 2, 0)
                # .view(-1, self.kernel_out_channels)
                # for lvl in range(num_levels)
                mlvl_kernel_preds[lvl][img_id]
                .permute(1, 2, 0)
                .reshape(-1, self.kernel_out_channels)
                for lvl in range(num_levels)
            ]
            img_kernel_pred = torch.cat(img_kernel_pred, dim=0)
            img_kernel_pred_list.append(img_kernel_pred)
        img_kernel_preds = torch.stack(img_kernel_pred_list, dim=0)
        return img_kernel_preds

    def forward(self, mlvl_kernel_preds):
        return self._get_img_kernel_preds(mlvl_kernel_preds)


class SOLOV2ClsPredsDecoder(torch.nn.Module):
    def __init__(self, num_classes):
        super(SOLOV2ClsPredsDecoder, self).__init__()
        self.cls_out_channels = num_classes

    def _get_img_cls_preds(self, mlvl_cls_scores):
        num_imgs = mlvl_cls_scores[0].shape[0]
        num_levels = len(mlvl_cls_scores)
        mlvl_cls_scores = list(mlvl_cls_scores)
        for lvl in range(num_levels):
            cls_scores = mlvl_cls_scores[lvl]
            cls_scores = cls_scores.sigmoid()
            local_max = F.max_pool2d(cls_scores, 2, stride=1, padding=1)
            keep_mask = local_max[:, :, :-1, :-1] == cls_scores
            cls_scores = cls_scores * keep_mask
            mlvl_cls_scores[lvl] = cls_scores.permute(0, 2, 3, 1)

        img_cls_pred_list = []
        for img_id in range(num_imgs):

            img_cls_pred = [
                # mlvl_cls_scores[lvl][img_id].view(-1, self.cls_out_channels)
                # for lvl in range(num_levels)
                mlvl_cls_scores[lvl][img_id].reshape(-1, self.cls_out_channels)
                for lvl in range(num_levels)
            ]
            img_cls_pred = torch.cat(img_cls_pred, dim=0)
            img_cls_pred_list.append(img_cls_pred)
        img_cls_preds = torch.stack(img_cls_pred_list, dim=0)
        return img_cls_preds

    def forward(self, mlvl_cls_scores):
        return self._get_img_cls_preds(mlvl_cls_scores)


class SimpleSOLOV2Decoder(torch.nn.Module):
    def __init__(
        self,
        num_classes,
        mask_stride,
        num_grids,
        strides,
        num_levels,
        dynamic_conv_size,
        test_cfg,
    ):
        super(SimpleSOLOV2Decoder, self).__init__()
        # from test_cfg
        # self.score_thr = test_cfg.score_thr
        # todo 固定得分阈值
        self.score_thr = 0.4
        self.mask_thr = test_cfg.mask_thr
        self.nms_pre = test_cfg.nms_pre
        self.max_per_img = test_cfg.max_per_img
        self.kernel = test_cfg.kernel
        self.sigma = test_cfg.sigma
        self.filter_thr = test_cfg.filter_thr

        self.cls_out_channels = num_classes
        self.mask_stride = mask_stride
        self.strides = strides
        self.num_grids = num_grids
        self.num_levels = num_levels
        self.dynamic_conv_size = dynamic_conv_size

    def _format_results(self):
        # 必须显式指定类型才能被scripts识别
        results_list: List[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []
        return results_list


    def _get_results(
        self, img_kernel_preds, img_cls_preds, mask_feats, img_h: int, img_w: int
    ):
        num_imgs = int(img_kernel_preds.shape[0])
        assert num_imgs == int(img_cls_preds.shape[0])

        # 必须显式指定类型才能被scripts识别
        results_list = self._format_results()

        for img_id in range(num_imgs):
            img_kernel_pred = img_kernel_preds[img_id]
            img_cls_pred = img_cls_preds[img_id]
            img_mask_feats = mask_feats[[img_id]]

            results = self._get_results_single(
                img_kernel_pred, img_cls_pred, img_mask_feats, img_h, img_w
            )

            results_list.append(results)

        return results_list

    def _empty_results(
        self, img_h: int, img_w: int, device: torch.device
    ):
        results_scores = torch.ones((0,), dtype=torch.float32).to(device)
        results_labels = torch.ones((0,), dtype=torch.int32).to(device)
        results_masks = torch.zeros((0, img_h, img_w), dtype=torch.bool).to(device)
        return results_scores, results_masks, results_labels


    def _get_results_single(
        self, kernel_preds, cls_scores, mask_feats, img_h: int, img_w: int
    ):
        # def empty_results(results, cls_scores):
        #     """Generate a empty results."""
        #     results.scores = cls_scores.new_ones(0)
        #     results.masks = cls_scores.new_zeros(0, *results.ori_shape[:2])
        #     results.labels = cls_scores.new_ones(0)
        #     return results

        # results_scores = cls_scores.new_ones(0)
        # results_masks = cls_scores.new_zeros(0, img_h, img_w)
        # results_labels = cls_scores.new_ones(0)


        # cfg = self.test_cfg if cfg is None else cfg
        # assert len(kernel_preds) == len(cls_scores)
        # results = InstanceData(img_meta)

        featmap_size = mask_feats.size()[-2:]

        img_shape = (img_h, img_w, 3)
        ori_shape = img_shape

        # overall info
        h, w, _ = img_shape
        upsampled_size = (
            featmap_size[0] * self.mask_stride,
            featmap_size[1] * self.mask_stride,
        )

        # process.
        # score_mask = (cls_scores > cfg.score_thr)
        score_mask = cls_scores > self.score_thr
        cls_scores = cls_scores[score_mask]
        if len(cls_scores) == 0:
            # return empty_results(results, cls_scores)
            return self._empty_results(
                img_h, img_w, cls_scores.device
            )

        # cate_labels & kernel_preds
        inds = score_mask.nonzero()
        cls_labels = inds[:, 1]
        #
        kernel_preds = kernel_preds[inds[:, 0]]


        # trans vector.
        # lvl_interval = cls_labels.new_tensor(self.num_grids).pow(2).cumsum(0)
        lvl_interval = (
            torch.tensor(self.num_grids, dtype=cls_labels.dtype)
            .to(cls_labels.device)
            .pow(2)
            .cumsum(0)
        )
        # strides = kernel_preds.new_ones(lvl_interval[-1])
        strides = torch.ones((lvl_interval[-1]), dtype=kernel_preds.dtype).to(
            kernel_preds.device
        )

        strides[: lvl_interval[0]] *= self.strides[0]
        for lvl in range(1, self.num_levels):
            strides[lvl_interval[lvl - 1] : lvl_interval[lvl]] *= self.strides[lvl]
        strides = strides[inds[:, 0]]

        # mask encoding.
        kernel_preds = kernel_preds.view(
            kernel_preds.size(0), -1, self.dynamic_conv_size, self.dynamic_conv_size
        )
        mask_preds = F.conv2d(mask_feats, kernel_preds, stride=1).squeeze(0).sigmoid()


        # mask.
        # masks = mask_preds > cfg.mask_thr
        masks = mask_preds > self.mask_thr
        sum_masks = masks.sum((1, 2)).float()
        keep = sum_masks > strides
        if keep.sum() == 0:
            # return empty_results(results, cls_scores)
            return self._empty_results(
                img_h, img_w, cls_scores.device
            )
        masks = masks[keep]
        mask_preds = mask_preds[keep]
        sum_masks = sum_masks[keep]
        cls_scores = cls_scores[keep]
        cls_labels = cls_labels[keep]




        # maskness.
        mask_scores = (mask_preds * masks).sum((1, 2)) / sum_masks
        cls_scores *= mask_scores

        scores, labels, _, keep_inds = mask_matrix_nms(
            masks,
            cls_labels,
            cls_scores,
            mask_area=sum_masks,
            # nms_pre=cfg.nms_pre,
            # max_num=cfg.max_per_img,
            # kernel=cfg.kernel,
            # sigma=cfg.sigma,
            # filter_thr=cfg.filter_thr,
            nms_pre=self.nms_pre,
            max_num=self.max_per_img,
            kernel=self.kernel,
            sigma=self.sigma,
            filter_thr=self.filter_thr,
        )

        # fix error
        if keep_inds.shape[0] == 0:
            return self._empty_results(
                img_h, img_w, cls_scores.device
            )
        mask_preds = mask_preds[keep_inds]

        mask_preds = F.interpolate(
            mask_preds.unsqueeze(0),
            size=upsampled_size,
            mode="bilinear",
            align_corners=False,
        )[:, :, :h, :w]
        # mask_preds = F.interpolate(
        #     mask_preds, size=ori_shape[:2], mode="bilinear", align_corners=False
        # ).squeeze(0)
        mask_preds = F.interpolate(
            mask_preds,
            size=(int(ori_shape[0]), int(ori_shape[1])),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        # masks = mask_preds > cfg.mask_thr
        masks = mask_preds > self.mask_thr

        # results.masks = masks
        # results.labels = labels
        # results.scores = scores
        # return results

        results_scores = scores
        results_masks = masks
        results_labels = labels

        # filter some cls
        keep = results_labels == 0
        results_scores = results_scores[keep]
        results_masks = results_masks[keep]
        results_labels = results_labels[keep]

        return results_scores, results_masks, results_labels

    def forward(
        self, img_kernel_preds, img_cls_preds, mask_feats, img_h: int, img_w: int
    ):
        return self._get_results(
            img_kernel_preds, img_cls_preds, mask_feats, img_h, img_w
        )

class SimpleSOLOV2DecoderNoPost(SimpleSOLOV2Decoder):
    def _empty_results(
        self, img_h: int, img_w: int, device: torch.device
    ):
        results_scores = torch.ones((0,), dtype=torch.float32).to(device)
        results_labels = torch.ones((0,), dtype=torch.int32).to(device)
        results_masks_pred = torch.zeros((0, img_h, img_w), dtype=torch.float32).to(device)
        strides = torch.ones((0,), dtype=torch.int32).to(device)
        return results_scores, results_masks_pred, results_labels, strides

    def _format_results(self):
        # 必须显式指定类型才能被scripts识别
        results_list: List[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []
        return results_list

    def _get_results_single(
        self, kernel_preds, cls_scores, mask_feats, img_h: int, img_w: int
    ):

        featmap_size = mask_feats.size()[-2:]

        img_shape = (img_h, img_w, 3)
        ori_shape = img_shape

        # overall info
        h, w, _ = img_shape
        upsampled_size = (
            featmap_size[0] * self.mask_stride,
            featmap_size[1] * self.mask_stride,
        )

        # process.
        # score_mask = (cls_scores > cfg.score_thr)
        score_mask = cls_scores > self.score_thr
        cls_scores = cls_scores[score_mask]
        if len(cls_scores) == 0:
            # return empty_results(results, cls_scores)
            return self._empty_results(
                img_h, img_w, cls_scores.device
            )

        # cate_labels & kernel_preds
        inds = score_mask.nonzero()
        cls_labels = inds[:, 1]
        kernel_preds = kernel_preds[inds[:, 0]]

        # trans vector.
        # lvl_interval = cls_labels.new_tensor(self.num_grids).pow(2).cumsum(0)
        lvl_interval = (
            torch.tensor(self.num_grids, dtype=cls_labels.dtype)
            .to(cls_labels.device)
            .pow(2)
            .cumsum(0)
        )
        # strides = kernel_preds.new_ones(lvl_interval[-1])
        strides = torch.ones((lvl_interval[-1]), dtype=kernel_preds.dtype).to(
            kernel_preds.device
        )

        strides[: lvl_interval[0]] *= self.strides[0]
        for lvl in range(1, self.num_levels):
            strides[lvl_interval[lvl - 1] : lvl_interval[lvl]] *= self.strides[lvl]
        strides = strides[inds[:, 0]]

        # mask encoding.
        kernel_preds = kernel_preds.view(
            kernel_preds.size(0), -1, self.dynamic_conv_size, self.dynamic_conv_size
        )
        mask_preds = F.conv2d(mask_feats, kernel_preds, stride=1).squeeze(0).sigmoid()

        return cls_scores, mask_preds, cls_labels, strides



def preprocess(
    img,
    input_shape_hw=(544, 1120),
    img_mean=(123.675, 116.28, 103.53),
    img_std=(58.395, 57.12, 57.375),
    to_rgb=True,
):
    img_resize = cv2.resize(img, (input_shape_hw[1], input_shape_hw[0]))
    if to_rgb:
        cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
    img_resize = img_resize.astype(np.float32)
    for i, (m, s) in enumerate(zip(img_mean, img_std)):
        img_resize[:, :, i] = (img_resize[:, :, i] - m) / s

    return img_resize


def infer_by_libtorch_module(img, model, device):
    img_preprocess = preprocess(
        img,
        input_shape_hw=input_shape_hw,
        img_mean=(123.675, 116.28, 103.53),
        img_std=(58.395, 57.12, 57.375),
        to_rgb=True,
    )
    # single image to tensor
    img_tensor = torch.tensor(img_preprocess[None], dtype=torch.float32).permute(
        0, 3, 1, 2
    )

    # to device
    img_tensor = img_tensor.to(device)

    # infer to get results
    results = model(img_tensor)

    # for result in results:
    #     scores, masks, labels = result

    return results




if __name__ == "__main__":
    from mmcv import Config
    from mmdet.apis.inference import init_detector, inference_detector
    from custom_tools.vis import draw_instance_seg_results
    from custom_tools.cvutils import show_image, read_image
    import cv2
    from pathlib import Path

    cfg_path = r"D:\project\1_qiti\model\20220922\solov2_light_r50_qiti.py"
    checkpoint = r"D:\project\1_qiti\model\20220922\qiti_anti_qiangban.pth"
    jit_module_save_path = str(Path(checkpoint).with_suffix(".pt"))
    example_img_path = r"D:\project\0_qiangban\dataset\images\巴城项目-7栋栋-4层-初测-砌筑工程-巡检（墙板+砌体）\train\BZL_20220807163927_4864_7栋D4F_FocusS70LLS082119198_2c352225-94e2-4e92-9671-402b4eb4e15d_025.png"
    device = "cpu"
    # device = "cuda"

    # input_shape_hw = (544, 1120)
    input_shape_hw = (544, 1312)

    # build origin model
    cfg = Config.fromfile(cfg_path)
    model = init_detector(cfg, checkpoint, device)

    # get example
    img = read_image(example_img_path, cv2.IMREAD_COLOR)
    # results = inference_detector(model, img)


    # build libtorch model
    new_model = SimpleSOLOV2.build_from_mmdet_module(
        model,
        input_shape_hw=input_shape_hw,
        device=device,
        use_postprocess=True,
    )
    new_model.to(device).eval()

    model_script = torch.jit.script(new_model)
    model_script.to(device).eval()

    # save module
    model_script.save(jit_module_save_path)
    print(f"save script model to {jit_module_save_path}.")
    exit()

    # load module
    model_script = torch.jit.load(jit_module_save_path).to(device).eval()

    # test
    # img = read_image(r"D:\project\1_qiti\dataset\images\9_9_add_11_17_new_all_origin\BZL_20210529161148_031_6D23F_LLS082017950 2_4d6d7e7b-058a-4768-897c-be4ac51cbe62_018_srcImage.png")
    results = infer_by_libtorch_module(img, model_script, device)
    scores, masks, labels = results[0]
    print(masks.shape)
    masks = masks.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    # resize masks
    keep = scores > 0.4
    masks = list(masks[keep])
    labels = list(labels[keep])
    scores = list(scores[keep])
    for i in range(len(masks)):
        masks[i] = (
            cv2.resize(
                masks[i].astype(np.uint8),
                (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        )



    vis = draw_instance_seg_results(img, masks, labels, scores)
    show_image(vis)
