import torch
import torch.nn as nn
from torch.nn.functional import mse_loss


class MSExp(nn.Module):

    def __init__(self, base: float):
        super().__init__()
        self.base: float = base

    def forward(self, inputs, targets):
        base_tensor = torch.full([len(inputs)], self.base)
        exp_loss = torch.sub(torch.sum(torch.pow(base_tensor, torch.abs(targets - inputs))), 1)
        return mse_loss(inputs, targets) + exp_loss
