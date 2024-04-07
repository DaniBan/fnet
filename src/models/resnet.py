import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_channels=in_planes,
                     out_channels=out_planes,
                     kernel_size=1,
                     stride=stride)


def conv3x3(in_planes, out_planes, stride=1, padding=1):
    return nn.Conv2d(in_channels=in_planes,
                     out_channels=out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=padding)


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, padding=1):
        super().__init__()
        # 1x1 conv, 64 (planes)
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        # 3x3 conv, 64 (planes)
        self.conv2 = conv3x3(planes, planes, stride, padding)
        self.bn2 = nn.BatchNorm2d(planes)
        # 1x1 conv, 256 (planes * expansion)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        if in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=2),
                nn.BatchNorm2d(planes * self.expansion)
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out
