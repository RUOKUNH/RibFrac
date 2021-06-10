import torch.nn as nn
import torch.nn.functional as F
import torch

class SoftDiceLoss(nn.Module):
    def __init__(self, weight = None, size_average = True, alpha = 1, gamma = 0.3):
        super(SoftDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1
        
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = m1 * m2
        # print("inter sum", intersection.sum(1))
        # print("prob sum", m1.sum(1))
        print("label sum", m2.sum(1))
        
        score = (2. * intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        
        ce = F.binary_cross_entropy_with_logits(logits, targets)
        fc = self.alpha * (1 - torch.exp(-ce)) ** self.gamma * ce
        
        loss = (score + fc) / 2
        
        return loss
        