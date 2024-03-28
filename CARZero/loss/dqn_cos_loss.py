## 使用 nn.BCEWithLogitsLoss() 作为loss， 对于label为-1的值，不计算loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import ipdb

class DQNCOSLoss(nn.Module):
    def __init__(self):
        super(DQNCOSLoss, self).__init__()

    def forward(self, input):
        batch_size = input.size(0)
        target = Variable(torch.LongTensor(range(batch_size))).to(input.device)
        loss = 0
        loss += nn.CrossEntropyLoss()(input, target)
        loss += nn.CrossEntropyLoss()(input.transpose(1, 0), target)
        return loss / 2