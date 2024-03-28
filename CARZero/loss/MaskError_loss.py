## 使用 nn.BCEWithLogitsLoss() 作为loss， 对于label为-1的值，不计算loss

import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

class MaskErrorLoss(nn.Module):
    def __init__(self):
        super(MaskErrorLoss, self).__init__()

    def forward(self, input, target):
        mask = (target != -1).float()    
        # loss = F.smooth_l1_loss(F.tanh(input), target, reduction="none")
        loss = F.l1_loss(F.tanh(input), target, reduction="none")
        # loss = F.mse_loss(F.tanh(input), target, reduction="none")
        loss = loss * mask
        return loss.sum() / mask.sum()