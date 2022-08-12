import torch
import torch.nn as nn

from models.utils import ConvBlock

CONV_LAYERS = [
    ("conv", 96, 7, 2, 2),  # type, filters, kernel, stride, padding
    ("maxpool", None, 3, 2, 0),
    ("conv", 256, 5, 2, 0),
    ("maxpool", None, 3, 2, 1),
    ("conv", 384, 3, 1, 1),
    ("conv", 384, 3, 1, 1),
    ("conv", 256, 3, 1, 1),
    ("maxpool", None, 3, 2, 0),
]


class ZFNet(nn.Module):
    def __init__(self, class_numbers, in_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.class_numbers = class_numbers
        self.conv_layers = self._create_conv_layers()
        self.fcs = self._create_fcs()

    def forward(self, x):
        x = self.conv_layers(x)
        return self.fcs(x)

    def _create_conv_layers(self):
        layers = []
        in_channels = self.in_channels
        for layer_info in CONV_LAYERS:
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

    def _create_fcs(self):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(9216, 4096),
            nn.Dropout(),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, 4096),
            nn.Dropout(),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, self.class_numbers),
            nn.Softmax(dim=1)
        )


def verify_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_example = torch.rand(8, 3, 224, 224).to(device)
    model = ZFNet(class_numbers=4).to(device)
    out = model(input_example)
    print(out)
    print(out.shape)


if __name__ == '__main__':
    verify_model()
