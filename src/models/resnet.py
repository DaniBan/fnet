import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, padding=1):
        super().__init__()

        # Block 1
        self.conv1 = conv3x3(in_planes=in_planes, out_planes=planes, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(planes)
        # Block 2
        self.conv2 = conv3x3(in_planes=planes, out_planes=planes, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm2d(planes)

        if in_planes != planes:
            self.downsample = conv1x1(in_planes=in_planes, out_planes=planes, stride=stride)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.bn1(self.conv1(x))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


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

        if in_planes != planes * self.expansion and stride != 1:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes * self.expansion, stride=2),
                nn.BatchNorm2d(planes * self.expansion)
            )
        elif in_planes == planes:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes * self.expansion, stride=1),
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
        out = F.relu(out)

        return out


class ResNetLayer(nn.Sequential):
    def __init__(self, in_channels, channels, num_blocks, stride=1, padding=1, blocktype="bottleneck", first=False):

        if blocktype == "bottleneck":
            block = BottleneckBlock
        elif blocktype == "basic":
            raise ValueError(f"Block type <{blocktype}> not implemented")
        else:
            raise ValueError("Unknown block type")

        if first:
            layers = [block(in_planes=channels, planes=channels, stride=1, padding=padding)]
        else:
            layers = [block(in_planes=in_channels, planes=channels, stride=stride, padding=padding)]

        for _ in range(1, num_blocks):
            layers.append(block(in_planes=channels * block.expansion, planes=channels, stride=1, padding=padding))

        self.in_channels = in_channels
        self.out_channels = channels * block.expansion
        super().__init__(*layers)


class ResNet(nn.Module):
    count = 0

    def __init__(self, num_blocks):
        super().__init__()

        # common layers
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # scale specific layers
        self.layer_2 = nn.Sequential(
            ResNetLayer(128, 64, num_blocks[0], stride=2, padding=1, first=True),
            ResNetLayer(256, 128, num_blocks[1], stride=2, padding=1),
            ResNetLayer(512, 256, num_blocks[2], stride=2, padding=1),
            ResNetLayer(1024, 512, num_blocks[3], stride=2, padding=1)
        )

        self.layer_3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=2048 * 8 * 8, out_features=18)
        )

    def forward(self, x):
        self.count += 1
        return self.layer_3(self.layer_2(self.layer_1(x)))


def res_net_50():
    num_blocks = [3, 4, 6, 3]
    return ResNet(num_blocks=num_blocks)


if __name__ == '__main__':
    model = res_net_50()
    print("=======================================")
    summary(model, (3, 250, 250), batch_size=1)
