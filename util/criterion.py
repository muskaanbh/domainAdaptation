import torch
import torch.nn as nn
import torch.nn.functional as F


def kd_loss(outputs, targets, alpha, T):
  _,big_idx_target = torch.max(targets.data,dim=-1)
  KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),F.softmax(targets/T, dim=1)) * (alpha * T * T) + F.cross_entropy(outputs, big_idx_target) * (1. - alpha)
  return KD_loss
