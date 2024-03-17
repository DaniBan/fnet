from torch import Tensor
from torch import nn
from collections import OrderedDict


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
            ("conv_2_1", nn.Conv2d(in_channels=input_shape,
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
            ("linear", nn.Linear(in_features=0,
                                 out_features=output_shape))
        ]))

    def forward(self, x: Tensor) -> Tensor:
        x = self.feat_extractor(x)
        x = self.classifier(x)
        return x
