import torch
from torch.nn.functional import mse_loss


def mse_log_loss(logits, labels):
    return mse_loss(input=logits, target=labels) + torch.log(torch.abs(labels - input))
