import numpy as np

import torch
import torch.nn.functional as F
from typing import List

class SingleStageDetectorLibtorch(torch.nn.Module):
    def __init__(self, backbone, neck, bbox_head):
        super(SingleStageDetectorLibtorch, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.bbox_head = bbox_head

    def forward(self, img_bchw):
        feats = self.backbone(img_bchw)
        if self.neck is not None:
            feats = self.neck(feats)
        results = self.bbox_head(feats)
        return results




