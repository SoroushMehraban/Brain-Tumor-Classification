import torch
import torch.nn as nn

from models.utils import ConvBlock

STEM_LAYERS = [
    ("conv", 64, 7, 2, 3),  # type, filters, kernel, stride, padding
    ("maxpool", None, 3, 2, 1),
    ("conv", 192, 3, 1, 1),
    ("maxpool", None, 3, 2, 1),
]

INCEPTION_LAYERS = [
    ("inception", 64, 96, 128, 16, 32, 32),  # type, #1x1, #3x3 reduce, #3x3, #5x5 reduce, #5x5, pool_proj
    ("inception", 128, 128, 192, 32, 96, 64),
    ("maxpool", 3, 2, 1),  # type, kernel, stride, padding
    ("inception", 192, 96, 208, 16, 48, 64),
    ("inception", 160, 112, 224, 24, 64, 64),
    ("inception", 128, 128, 256, 24, 64, 64),
    ("inception", 112, 144, 288, 32, 64, 64),
    ("inception", 256, 160, 320, 32, 128, 128),
    ("maxpool", 3, 2, 1),
    ("inception", 256, 160, 320, 32, 128, 128),
    ("inception", 384, 192, 384, 48, 128, 128),
]


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_1x1, out_3x3_reduce, out_3x3, out_5x5_reduce, out_5x5, out_pool_proj):
        super().__init__()

        self.branch_1x1 = ConvBlock(in_channels, out_1x1, kernel_size=1, stride=1, padding=0)

        self.branch_3x3 = nn.Sequential(
            ConvBlock(in_channels, out_3x3_reduce, kernel_size=1, stride=1, padding=0),
            ConvBlock(out_3x3_reduce, out_3x3, kernel_size=3, stride=1, padding=1)
        )

        self.branch_5x5 = nn.Sequential(
            ConvBlock(in_channels, out_5x5_reduce, kernel_size=1, stride=1, padding=0),
            ConvBlock(out_5x5_reduce, out_5x5, kernel_size=5, stride=1, padding=2)
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, out_pool_proj, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        return torch.cat(
            [self.branch_1x1(x), self.branch_3x3(x), self.branch_5x5(x), self.branch_pool(x)],
            dim=1
        )


class GoogLeNet(nn.Module):
    def __init__(self, class_numbers, in_channels=3):
        super().__init__()
        self.in_channels = in_channels

        self.stem_layers = self._create_stem_layers()
        self.inception_layers = self._create_inception_layers()
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.dropout = nn.Dropout(p=0.4)
        self.linear = nn.Linear(1024, class_numbers)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.stem_layers(x)
        x = self.inception_layers(x)
        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.linear(x)
        return self.softmax(x)

    @staticmethod
    def _create_inception_layers():
        layers = []
        in_channels = 192
        for layer_info in INCEPTION_LAYERS:
            layer_type = layer_info[0]
            if layer_type == "inception":
                out_1x1, out_3x3_reduce, out_3x3, out_5x5_reduce, out_5x5, out_pool_proj = layer_info[1:]
                layers += [InceptionBlock(in_channels, out_1x1, out_3x3_reduce, out_3x3, out_5x5_reduce, out_5x5,
                                          out_pool_proj)]
                in_channels = out_1x1 + out_3x3 + out_5x5 + out_pool_proj
            elif layer_type == "maxpool":
                kernel_size, stride, padding = layer_info[1:]
                layers += [nn.MaxPool2d(kernel_size, stride, padding)]
            else:
                raise Exception("Invalid layer type")

        return nn.Sequential(*layers)

    def _create_stem_layers(self):
        layers = []
        in_channels = self.in_channels
        for layer_info in STEM_LAYERS:
            layer_type, filter_numbers, kernel_size, stride, padding = layer_info
            if layer_type == "conv":
                layers += [
                    ConvBlock(in_channels, filter_numbers, kernel_size, stride, padding)
                ]
                in_channels = filter_numbers
            elif layer_type == "maxpool":
                layers += [
                    nn.MaxPool2d(kernel_size, stride, padding)
                ]
            else:
                raise Exception("Invalid layer type")

        return nn.Sequential(*layers)


def verify_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_example = torch.rand(4, 3, 224, 224).to(device)
    model = GoogLeNet(class_numbers=4).to(device)
    out = model(input_example)
    print(out.shape)


if __name__ == '__main__':
    verify_model()
