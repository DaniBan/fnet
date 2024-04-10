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


class ResNetLayer(nn.Sequential):
    def __init__(self, in_channels, channels, num_blocks, stride=1, padding=1, blocktype="bottleneck"):

        if blocktype == "bottleneck":
            block = BottleneckBlock
        elif blocktype == "basic":
            raise ValueError(f"Block type <{blocktype}> not implemented")
        else:
            raise ValueError("Unknown block type")

        layers = [block(in_planes=in_channels, planes=channels, stride=stride, padding=padding)]
        for _ in range(1, num_blocks):
            layers.append(block(in_planes=in_channels, planes=channels, stride=1, padding=padding))

        self.in_channels = in_channels
        self.out_channels = channels * BottleneckBlock.expansion
        super(ResNetLayer).__init__(*layers)


class ResNet(nn.Module):

    def __init__(self, num_blocks):
        super().__init__()

        # common layers
        layer_1 = ResNetLayer(64, 64, num_blocks[0], stride=1, padding=1)
        layer_2 = ResNetLayer(256, 128, num_blocks[0], stride=1, padding=1)
        layer_3 = ResNetLayer(512, 256, num_blocks[0], stride=1, padding=1)
        layer_5 = ResNetLayer(1024, 512, num_blocks[0], stride=1, padding=1)

        # scale specific layers

        print("Not implemented")


def res_net_50():
    num_blocks = [3, 4, 6, 3]
    return ResNet(num_blocks=num_blocks)
