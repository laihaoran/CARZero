## 使用 nn.BCEWithLogitsLoss() 作为loss， 对于label为-1的值，不计算loss

import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

class MaskBCELoss(nn.Module):
    def __init__(self):
        super(MaskBCELoss, self).__init__()

    def forward(self, input, target):
        mask = (target != -1).float()
        
        loss = F.binary_cross_entropy_with_logits(input, target, reduction="none")
        loss = loss * mask
        return loss.sum() / mask.sum()