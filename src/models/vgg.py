from collections import OrderedDict

from torch import nn


class VGG16(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()

        self.feat_extractor = nn.Sequential(OrderedDict([
            # Block 1
            ("conv_1_1", nn.Conv2d(in_channels=input_shape, out_channels=64, kernel_size=3, padding=1)),
            ("relu_1_1", nn.ReLU()),
            ("conv_1_2", nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)),
            ("relu_1_2", nn.ReLU()),
            ("max_pool_1", nn.MaxPool2d(2, stride=2)),
            # 125
            # Block 2
            ("conv_2_1", nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)),
            ("relu_2_1", nn.ReLU()),
            ("conv_2_2", nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)),
            ("relu_2_2", nn.ReLU()),
            ("max_pool_2", nn.MaxPool2d(2, stride=2)),
            # 62
            # Block 3
            ("conv_3_1", nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)),
            ("relu_3_1", nn.ReLU()),
            ("conv_3_2", nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)),
            ("relu_3_2", nn.ReLU()),
            ("conv_3_3", nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)),
            ("relu_3_3", nn.ReLU()),
            ("max_pool_3", nn.MaxPool2d(2, stride=2)),
            # 31
            # Block 4
            ("conv_4_1", nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)),
            ("relu_4_1", nn.ReLU()),
            ("conv_4_2", nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)),
            ("relu_4_2", nn.ReLU()),
            ("conv_4_3", nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)),
            ("relu_4_3", nn.ReLU()),
            ("max_pool_4", nn.MaxPool2d(2, stride=2)),
            # 15
            # Block 5
            ("conv_5_1", nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)),
            ("relu_5_1", nn.ReLU()),
            ("conv_5_2", nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)),
            ("relu_5_2", nn.ReLU()),
            ("conv_5_3", nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)),
            ("relu_5_3", nn.ReLU()),
            ("max_pool_5", nn.MaxPool2d(2, stride=2))
            # 7
        ]))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512 * 7 * 7, out_features=4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=output_shape)
        )

    def forward(self, x):
        x = self.feat_extractor(x)
        x = self.classifier(x)
        return x


class TinyVGG(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()

        self.feat_extractor = nn.Sequential(OrderedDict([
            # Block 1
            ("conv_1_1", nn.Conv2d(in_channels=input_shape,
                                   out_channels=hidden_units,
                                   kernel_size=3,
                                   padding=1)),
            ("relu_1_1", nn.ReLU()),
            ("conv_1_2", nn.Conv2d(in_channels=hidden_units,
                                   out_channels=hidden_units,
                                   kernel_size=3,
                                   padding=1)),
            ("relu_1_2", nn.ReLU()),
            ("max_pool_1", nn.MaxPool2d(2)),

            # Block 2
            ("conv_2_1", nn.Conv2d(in_channels=hidden_units,
                                   out_channels=hidden_units,
                                   kernel_size=3,
                                   padding=1)),
            ("relu_2_1", nn.ReLU()),

            ("conv_2_2", nn.Conv2d(in_channels=hidden_units,
                                   out_channels=hidden_units,
                                   kernel_size=3,
                                   padding=1)),
            ("relu_2_2", nn.ReLU()),
            ("max_pool_2", nn.MaxPool2d(2))
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ("flatten", nn.Flatten()),
            ("linear", nn.Linear(in_features=3844 * hidden_units,
                                 out_features=output_shape))
        ]))

    def forward(self, x):
        x = self.feat_extractor(x)
        x = self.classifier(x)
        return x
