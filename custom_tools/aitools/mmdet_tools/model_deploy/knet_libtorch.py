from pathlib import Path
import torch.nn.functional as F
from typing import List, Tuple
from deploy._test_utils import infer_by_libtorch_module, preprocess

import cv2
import numpy as np

mmdet_dir_parent = str(Path(__file__).parent.parent)
import sys
sys.path = [mmdet_dir_parent] + sys.path
sys.path.append(str(Path(mmdet_dir_parent) / "KNet_main"))
import torch
torch.autograd.set_detect_anomaly(True)
from torch import nn
from mmdet.models import HEADS
ConvKernelHead = HEADS.get("ConvKernelHead")
# from KNet_main.knet.det.kernel_head import ConvKernelHead

class KNetInference(nn.Module):
    def __init__(self, backbone, neck, rpn_head, roi_head, decoder):
        super(KNetInference, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.rpn_head = rpn_head
        self.roi_head = roi_head
        self.decoder = decoder

    def forward(self, x):
        feats = self.backbone(x)
        if self.neck is not None:
            feats = self.neck(feats)
        proposal_feats, x_feats, mask_preds, seg_preds = self.rpn_head(feats)
        cls_scores, scaled_mask_preds = self.roi_head(x_feats, proposal_feats, mask_preds)
        results = self.decoder(cls_scores, scaled_mask_preds)
        return results

    @classmethod
    def build_from_mmdet_module(
        cls, model, inputs, device="cuda:0"
    ):
        model.to(device).eval()
        b, c, h, w = inputs.shape
        x = inputs.to(device)
        # if img is None:
        #     x = torch.rand((b, 3, h, w), dtype=torch.float32).to(device)
        # else:
        #     x = cv2.resize(img, dsize=(w, h))
        #     x = [x for _ in range(batch_size)]
        #     x = (torch.permute(torch.tensor(x, dtype=torch.float32), dims=(0, 3, 1, 2)) / 255).to(device)

        # backbone
        backbone = torch.jit.trace(model.backbone, x, strict=False)
        feats = backbone(x)

        # neck
        if model.with_neck:
            assert model.neck.__class__.__name__ == "FPN"
            neck = torch.jit.trace(model.neck, (feats, ), strict=False)
            feats = neck(feats)
        else:
            neck = None

        # rpn
        assert model.rpn_head.__class__.__name__ == "ConvKernelHead"
        rpn_head = torch.jit.trace(ConvKernelHeadInference(model.rpn_head), example_inputs=(feats, ))
        proposal_feats, x_feats, mask_preds, seg_preds = rpn_head(feats)

        # roi
        assert model.roi_head.__class__.__name__ == "KernelIterHead"
        mask_head_list = nn.ModuleList()
        for mask_head in model.roi_head.mask_head:
            assert mask_head.__class__.__name__ == "KernelUpdateHead"
            mask_head_list.append(KernelUpdateHeadInference(mask_head))
        roi_head = torch.jit.trace(KernelIterHeadInference(mask_head_list), example_inputs=(x_feats, proposal_feats, mask_preds))
        cls_scores, scaled_mask_preds = roi_head(x_feats, proposal_feats, mask_preds)

        # decoder
        if model.roi_head.do_panoptic:
            raise NotImplementedError
        else:
            # decoder = KNetDecoderNotDoPanoptic()
            decoder_ori = KNetDecoderNotDoPanoptic(
                num_classes=model.roi_head.mask_head[-1].num_classes,
                max_per_img=model.roi_head.test_cfg.max_per_img,
                mask_thr=model.roi_head.mask_head[-1].mask_thr,
                ori_h=h,
                ori_w=w,
            )
            decoder = torch.jit.script(decoder_ori)
            results = decoder(cls_scores, scaled_mask_preds)
        return cls(backbone, neck, rpn_head, roi_head, decoder)

class ConvKernelHeadInference(nn.Module):
    def __init__(self, conv_kernel_head):
        super(ConvKernelHeadInference, self).__init__()
        # self.conv_kernel_head = conv_kernel_head
        self.localization_fpn = conv_kernel_head.localization_fpn
        self.loc_convs = conv_kernel_head.loc_convs
        self.feat_downsample_stride = conv_kernel_head.feat_downsample_stride
        self.feat_refine = conv_kernel_head.feat_refine
        if self.feat_downsample_stride > 1 and self.feat_refine:
            self.ins_downsample = conv_kernel_head.ins_downsample
            self.seg_downsample = conv_kernel_head.seg_downsample
        self.init_kernels = conv_kernel_head.init_kernels
        self.semantic_fpn = conv_kernel_head.semantic_fpn
        self.seg_convs = conv_kernel_head.seg_convs
        self.proposal_feats_with_obj = conv_kernel_head.proposal_feats_with_obj
        self.cat_stuff_mask = conv_kernel_head.cat_stuff_mask
        self.use_binary = conv_kernel_head.use_binary
        self.num_proposals = conv_kernel_head.num_proposals
        self.out_channels = conv_kernel_head.out_channels
        self.conv_seg = conv_kernel_head.conv_seg
        self.num_thing_classes = conv_kernel_head.num_thing_classes

    @staticmethod
    # @torch.jit.script_if_tracing
    @torch.jit._script_if_tracing
    def get_mask(pred):
        return pred > 0.5

    def forward(self, feats):
        # torch scripts追踪输入不能为"Tuple[Tuple[Tensor, Tensor, Tensor, Tensor], Tuple[int]]"
        num_imgs = feats[0].shape[0]

        localization_feats = self.localization_fpn(feats)
        if isinstance(localization_feats, list):
            loc_feats = localization_feats[0]
        else:
            loc_feats = localization_feats
        for conv in self.loc_convs:
            loc_feats = conv(loc_feats)
        if self.feat_downsample_stride > 1 and self.feat_refine:
            loc_feats = self.ins_downsample(loc_feats)
        mask_preds = self.init_kernels(loc_feats)

        if self.semantic_fpn:
            if isinstance(localization_feats, list):
                semantic_feats = localization_feats[1]
            else:
                semantic_feats = localization_feats
            for conv in self.seg_convs:
                semantic_feats = conv(semantic_feats)
            if self.feat_downsample_stride > 1 and self.feat_refine:
                semantic_feats = self.seg_downsample(semantic_feats)
        else:
            semantic_feats = None

        if semantic_feats is not None:
            seg_preds = self.conv_seg(semantic_feats)
        else:
            seg_preds = None

        proposal_feats = self.init_kernels.weight.clone()
        proposal_feats = proposal_feats[None].expand(num_imgs,
                                                     *proposal_feats.size())

        if semantic_feats is not None:
            x_feats = semantic_feats + loc_feats
        else:
            x_feats = loc_feats





        if self.proposal_feats_with_obj:
            sigmoid_masks = mask_preds.sigmoid()
            # nonzero_inds = sigmoid_masks > 0.5
            nonzero_inds = self.get_mask(sigmoid_masks)
            if self.use_binary:
                sigmoid_masks = nonzero_inds.float()
            else:
                sigmoid_masks = nonzero_inds.float() * sigmoid_masks
            obj_feats = torch.einsum('bnhw,bchw->bnc', sigmoid_masks, x_feats)

        # if self.proposal_feats_with_obj:
            proposal_feats = proposal_feats + obj_feats.view(
                num_imgs, self.num_proposals, self.out_channels, 1, 1)

        cls_scores = None

        if self.cat_stuff_mask and not self.training:
            mask_preds = torch.cat(
                [mask_preds, seg_preds[:, self.num_thing_classes:]], dim=1)
            stuff_kernels = self.conv_seg.weight[self.
                                                 num_thing_classes:].clone()
            stuff_kernels = stuff_kernels[None].expand(num_imgs,
                                                       *stuff_kernels.size())
            proposal_feats = torch.cat([proposal_feats, stuff_kernels], dim=1)

        # 返回值不允许存在None
        # return proposal_feats, x_feats, mask_preds, cls_scores, seg_preds
        return proposal_feats, x_feats, mask_preds, seg_preds

class KernelUpdateHeadInference(nn.Module):
    def __init__(self, kernel_update_head):
        super(KernelUpdateHeadInference, self).__init__()
        self.feat_transform = kernel_update_head.feat_transform
        self.hard_mask_thr = kernel_update_head.hard_mask_thr
        self.in_channels = kernel_update_head.in_channels
        self.kernel_update_conv = kernel_update_head.kernel_update_conv
        self.attention = kernel_update_head.attention
        self.attention_norm = kernel_update_head.attention_norm
        self.with_ffn = kernel_update_head.with_ffn
        self.ffn_norm = kernel_update_head.ffn_norm
        self.ffn = kernel_update_head.ffn
        self.cls_fcs = kernel_update_head.cls_fcs
        self.mask_fcs = kernel_update_head.mask_fcs
        self.fc_cls = kernel_update_head.fc_cls
        self.fc_mask = kernel_update_head.fc_mask
        self.mask_transform_stride = kernel_update_head.mask_transform_stride
        self.feat_gather_stride = kernel_update_head.feat_gather_stride
        self.conv_kernel_size = kernel_update_head.conv_kernel_size
        self.mask_upsample_stride = kernel_update_head.mask_upsample_stride
        self.use_sigmoid = kernel_update_head.loss_cls.use_sigmoid

    def forward(self,
                x,
                proposal_feat,
                mask_preds,
                # prev_cls_score=None,
                # mask_shape=None,
                # img_metas=None
                ):

        # prev_cls_score, img_metas不使用，去掉。
        # mask_shape不赋值，恒为None，去掉

        N, num_proposals = proposal_feat.shape[:2]
        if self.feat_transform is not None:
            x = self.feat_transform(x)
        C, H, W = x.shape[-3:]

        # mask_h, mask_w = mask_preds.shape[-2:]
        # if mask_h != H or mask_w != W:
        #     gather_mask = F.interpolate(
        #         mask_preds, (H, W), align_corners=False, mode='bilinear')
        # else:
        #     gather_mask = mask_preds

        gather_mask = F.interpolate(
            mask_preds, (H, W), align_corners=False, mode='bilinear')

        sigmoid_masks = gather_mask.sigmoid()
        nonzero_inds = sigmoid_masks > self.hard_mask_thr
        sigmoid_masks = nonzero_inds.float()

        # einsum is faster than bmm by 30%
        x_feat = torch.einsum('bnhw,bchw->bnc', sigmoid_masks, x)

        # obj_feat in shape [B, N, C, K, K] -> [B, N, C, K*K] -> [B, N, K*K, C]
        proposal_feat = proposal_feat.reshape(N, num_proposals,
                                              self.in_channels,
                                              -1).permute(0, 1, 3, 2)
        obj_feat = self.kernel_update_conv(x_feat, proposal_feat)

        # [B, N, K*K, C] -> [B, N, K*K*C] -> [N, B, K*K*C]
        obj_feat = obj_feat.reshape(N, num_proposals, -1).permute(1, 0, 2)
        obj_feat = self.attention_norm(self.attention(obj_feat))
        # [N, B, K*K*C] -> [B, N, K*K*C]
        obj_feat = obj_feat.permute(1, 0, 2)

        # obj_feat in shape [B, N, K*K*C] -> [B, N, K*K, C]
        obj_feat = obj_feat.reshape(N, num_proposals, -1, self.in_channels)

        # FFN
        if self.with_ffn:
            obj_feat = self.ffn_norm(self.ffn(obj_feat))

        cls_feat = obj_feat.sum(-2)
        mask_feat = obj_feat

        for cls_layer in self.cls_fcs:
            cls_feat = cls_layer(cls_feat)
        for reg_layer in self.mask_fcs:
            mask_feat = reg_layer(mask_feat)

        cls_score = self.fc_cls(cls_feat).view(N, num_proposals, -1)
        # [B, N, K*K, C] -> [B, N, C, K*K]
        mask_feat = self.fc_mask(mask_feat).permute(0, 1, 3, 2)

        if (self.mask_transform_stride == 2 and self.feat_gather_stride == 1):
            mask_x = F.interpolate(
                x, scale_factor=0.5, mode='bilinear', align_corners=False)
            H, W = mask_x.shape[-2:]
        else:
            mask_x = x
        # group conv is 5x faster than unfold and uses about 1/5 memory
        # Group conv vs. unfold vs. concat batch, 2.9ms :13.5ms :3.8ms
        # Group conv vs. unfold vs. concat batch, 278 : 1420 : 369
        # fold_x = F.unfold(
        #     mask_x,
        #     self.conv_kernel_size,
        #     padding=int(self.conv_kernel_size // 2))
        # mask_feat = mask_feat.reshape(N, num_proposals, -1)
        # new_mask_preds = torch.einsum('bnc,bcl->bnl', mask_feat, fold_x)
        # [B, N, C, K*K] -> [B*N, C, K, K]
        mask_feat = mask_feat.reshape(N, num_proposals, C,
                                      self.conv_kernel_size,
                                      self.conv_kernel_size)
        # [B, C, H, W] -> [1, B*C, H, W]
        new_mask_preds = []
        for i in range(N):
            new_mask_preds.append(
                F.conv2d(
                    mask_x[i:i + 1],
                    mask_feat[i],
                    padding=int(self.conv_kernel_size // 2)))

        new_mask_preds = torch.cat(new_mask_preds, dim=0)
        new_mask_preds = new_mask_preds.reshape(N, num_proposals, H, W)
        if self.mask_transform_stride == 2:
            new_mask_preds = F.interpolate(
                new_mask_preds,
                scale_factor=2,
                mode='bilinear',
                align_corners=False)

        # # mask_shape恒为None，去除
        # if mask_shape is not None and mask_shape[0] != H:
        #     new_mask_preds = F.interpolate(
        #         new_mask_preds,
        #         mask_shape,
        #         align_corners=False,
        #         mode='bilinear')

        return cls_score, new_mask_preds, obj_feat.permute(0, 1, 3, 2).reshape(
            N, num_proposals, self.in_channels, self.conv_kernel_size,
            self.conv_kernel_size)

class KernelIterHeadInference(nn.Module):
    def __init__(self, mask_head_list):
        super(KernelIterHeadInference, self).__init__()
        self.mask_head_list = mask_head_list


    def forward(self, x, proposal_feats, mask_preds):
        # 更改为将多个mask一同处理了
        object_feats = proposal_feats
        for stage, mask_head in enumerate(self.mask_head_list):
            cls_scores, mask_preds, object_feats = mask_head(x, object_feats, mask_preds)
            # if mask_head.mask_upsample_stride > 1 and (stage == self.num_stages - 1
            #                                            or self.training):
            #     scaled_mask_preds = F.interpolate(
            #         mask_preds,
            #         scale_factor=mask_head.mask_upsample_stride,
            #         align_corners=False,
            #         mode='bilinear')
            # else:
            #     scaled_mask_preds = mask_preds
        mask_head = self.mask_head_list[-1]
        if mask_head.mask_upsample_stride > 1:
            scaled_mask_preds = F.interpolate(
                mask_preds,
                scale_factor=mask_head.mask_upsample_stride,
                align_corners=False,
                mode='bilinear')
        else:
            scaled_mask_preds = mask_preds

        if self.mask_head_list[-1].use_sigmoid:
        # if self.mask_head_list[-1].loss_cls.use_sigmoid:
            cls_scores = cls_scores.sigmoid()
        else:
            cls_scores = cls_scores.softmax(-1)[..., :-1]

        return cls_scores, scaled_mask_preds

class KNetDecoderNotDoPanoptic(nn.Module):
    def __init__(self, num_classes, max_per_img, mask_thr, ori_h, ori_w):
        super(KNetDecoderNotDoPanoptic, self).__init__()
        self.num_classes = num_classes
        self.max_per_img = max_per_img
        self.mask_thr = mask_thr
        self.ori_h = ori_h
        self.ori_w = ori_w

    def rescale_masks(self, masks_per_img) -> torch.Tensor:
        # 修改为固定输出尺寸
        seg_masks = F.interpolate(
            masks_per_img.unsqueeze(0).sigmoid(),
            size=(self.ori_h, self.ori_w),
            mode='bilinear',
            align_corners=False).squeeze(0)
        return seg_masks

    # def segm2result(self, mask_preds, det_labels, cls_scores):
    #     num_classes = self.num_classes
    #     bbox_result = None
    #     segm_result = [[] for _ in range(num_classes)]
    #     # mask_preds = mask_preds.cpu().numpy()
    #     # det_labels = det_labels.cpu().numpy()
    #     # cls_scores = cls_scores.cpu().numpy()
    #     mask_preds = mask_preds.cpu()
    #     det_labels = det_labels.cpu()
    #     cls_scores = cls_scores.cpu()
    #     num_ins = mask_preds.shape[0]
    #     # fake bboxes
    #     bboxes = np.zeros((num_ins, 5), dtype=np.float32)
    #     bboxes[:, -1] = cls_scores
    #     bbox_result = [bboxes[det_labels == i, :] for i in range(num_classes)]
    #     for idx in range(num_ins):
    #         segm_result[det_labels[idx]].append(mask_preds[idx])
    #     return bbox_result, segm_result
    #
    # def get_seg_masks(self, masks_per_img, labels_per_img, scores_per_img):
    #     seg_masks = self.rescale_masks(masks_per_img)
    #     seg_masks = seg_masks > self.mask_thr
    #     bbox_result, segm_result = self.segm2result(seg_masks, labels_per_img, scores_per_img)
    #     return bbox_result, segm_result

    def forward(self, cls_scores, scaled_mask_preds):
        results_list: List[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []
        num_imgs = cls_scores.shape[0]
        for img_id in range(num_imgs):
            cls_score_per_img = cls_scores[img_id]
            scores_per_img, topk_indices = cls_score_per_img.flatten(
                0, 1).topk(
                self.max_per_img, sorted=True)
            mask_indices = topk_indices // self.num_classes
            labels_per_img = topk_indices % self.num_classes
            masks_per_img = scaled_mask_preds[img_id][mask_indices]
            # single_result = self.get_seg_masks(masks_per_img, labels_per_img, scores_per_img,)
            masks_per_img = self.rescale_masks(masks_per_img)
            masks_per_img = masks_per_img > self.mask_thr

            results_list.append((scores_per_img, masks_per_img, labels_per_img))
        return results_list



if __name__ == "__main__":
    from mmdet.apis.inference import init_detector, inference_detector
    from custom_tools.cvutils import read_image, show_image
    from custom_tools.vis import draw_instance_seg_results
    from pathlib import Path
    import mmcv

    # model_cfg_path = r"D:\project\7_liang_dong\model\knet_s3_r50_fpn_dp_1312x544_1205\knet_s3_r50_fpn_dp_1312x544.py"
    # checkpoint_path = r"D:\project\7_liang_dong\model\knet_s3_r50_fpn_dp_1312x544_1205\epoch_36.pth"
    # model_cfg_path = r"D:\project\7_liang_dong\model\circle_conv_knet_s3_r50_fpn_ldp_1312x544_depth_sobelxy_as12_1230\circle_conv_knet_s3_r50_fpn_ldp_1312x544.py"
    # checkpoint_path = r"D:\project\7_liang_dong\model\circle_conv_knet_s3_r50_fpn_ldp_1312x544_depth_sobelxy_as12_1230\epoch_54.pth"
    file_dir = r"D:\project\6_qiangban_qiti\model\knet_s3_r50_fpn_qiti_1312x544_0131"
    model_cfg_path = fr"{file_dir}\knet_s3_r50_fpn_qiti_1312x544.py"
    checkpoint_path = fr"{file_dir}\epoch_72.pth"
    save_torchscripts_path = str(Path(checkpoint_path).with_suffix(".pt"))
    # img_path = r"D:\project\7_liang_dong\dataset\images\old_20220427_0\bzl_Faro_scans_20201212092332_2800__dc86bcb0-8d6b-4e65-b918-32211d970bfb_024.png"
    # img_path = r"D:\project\7_liang_dong\dataset\depth_sobelxy_as12\old_20220427_0\bzl_Faro_scans_20201212092332_2800__dc86bcb0-8d6b-4e65-b918-32211d970bfb_024.png"
    img_path = r"D:\project\0_qiangban\dataset\images\巴城项目-7栋栋-4层-初测-砌筑工程-巡检（墙板+砌体）\train\BZL_20220807163927_4864_7栋D4F_FocusS70LLS082119198_2c352225-94e2-4e92-9671-402b4eb4e15d_025.png"
    # img_path = r"D:\project\0_qiangban\dataset\images\巴城项目-7栋栋-4层-初测-砌筑工程-巡检（墙板+砌体）\train\BZL_20220807163927_4864_7栋D4F_FocusS70LLS082119198_2c352225-94e2-4e92-9671-402b4eb4e15d_025.png"
    device = "cpu"
    # device = "cuda"
    input_shape_hw = (544, 1312)
    batch_size = 1

    img = read_image(img_path)

    convert = True

    if convert:

        model = init_detector(model_cfg_path, checkpoint=checkpoint_path)
        results = inference_detector(model, img)
        det_results, seg_results = results
        # from mmdet.apis import show_result_pyplot
        # show_result_pyplot(model, img, results)

        img_preprocess = preprocess(img, input_shape_hw=input_shape_hw)
        imgs_tensor = torch.from_numpy(np.repeat(img_preprocess[None], batch_size, axis=0)).permute(0, 3, 1, 2).to(torch.float32)

        model_inference = KNetInference.build_from_mmdet_module(
            model=model,
            inputs=imgs_tensor,
            device=device
        )

        # model_scripts = torch.jit.trace(model_inference, example_inputs=(imgs_tensor, ))
        model_scripts = torch.jit.script(model_inference)
        model_scripts.save(save_torchscripts_path)
        exit()

    model_scripts = torch.jit.load(save_torchscripts_path)
    #
    masks, labels, scores = infer_by_libtorch_module(
        img, model_scripts, device=device, input_shape_hw=input_shape_hw,
    )
    score_thr = 0.3
    keeps = [i for i, score in enumerate(scores) if score > score_thr]
    masks = [masks[i] for i in keeps]
    labels = [labels[i] for i in keeps]
    scores = [scores[i] for i in keeps]
    # img = np.full_like(img, fill_value=255)
    img_vis = draw_instance_seg_results(img, masks, labels, scores)
    show_image(img_vis)





