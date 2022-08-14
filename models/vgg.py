import torch
import torch.nn as nn

from models.utils import ConvBlock

VGG_16_CONV_LAYERS = [
    ("conv", 64, 3, 1, 1),  # type, filters, kernel, stride, padding
    ("conv", 64, 3, 1, 1),
    ("maxpool", None, 2, 2, 0),
    ("conv", 128, 3, 1, 1),
    ("conv", 128, 3, 1, 1),
    ("maxpool", None, 2, 2, 0),
    ("conv", 256, 3, 1, 1),
    ("conv", 256, 3, 1, 1),
    ("maxpool", None, 2, 2, 0),
    ("conv", 512, 3, 1, 1),
    ("conv", 512, 3, 1, 1),
    ("conv", 512, 3, 1, 1),
    ("maxpool", None, 2, 2, 0),
    ("conv", 512, 3, 1, 1),
    ("conv", 512, 3, 1, 1),
    ("conv", 512, 3, 1, 1),
    ("maxpool", None, 2, 2, 0),
]

VGG_19_CONV_LAYERS = [
    ("conv", 64, 3, 1, 1),  # type, filters, kernel, stride, padding
    ("conv", 64, 3, 1, 1),
    ("maxpool", None, 2, 2, 0),
    ("conv", 128, 3, 1, 1),
    ("conv", 128, 3, 1, 1),
    ("maxpool", None, 2, 2, 0),
    ("conv", 256, 3, 1, 1),
    ("conv", 256, 3, 1, 1),
    ("maxpool", None, 2, 2, 0),
    ("conv", 512, 3, 1, 1),
    ("conv", 512, 3, 1, 1),
    ("conv", 512, 3, 1, 1),
    ("conv", 512, 3, 1, 1),
    ("maxpool", None, 2, 2, 0),
    ("conv", 512, 3, 1, 1),
    ("conv", 512, 3, 1, 1),
    ("conv", 512, 3, 1, 1),
    ("conv", 512, 3, 1, 1),
    ("maxpool", None, 2, 2, 0),
]


class VGG(nn.Module):
    def __init__(self, class_numbers: int, version: int, in_channels=3):
        super().__init__()
        assert version in [16, 19]

        self.in_channels = in_channels
        self.class_numbers = class_numbers
        self.version = version
        self.conv_layers = self._create_conv_layers()
        self.fcs = self._create_fcs()

    def forward(self, x):
        x = self.conv_layers(x)
        return self.fcs(x)

    def _create_conv_layers(self):
        layers = []
        in_channels = self.in_channels

        conv_architecture = VGG_16_CONV_LAYERS if self.version == 16 else VGG_19_CONV_LAYERS
        for layer_info in conv_architecture:
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
            nn.Linear(25088, 4096),
            nn.Dropout(),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, 4096),
            nn.Dropout(),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, self.class_numbers)
        )


def verify_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== VGG16 ===")
    input_example = torch.rand(4, 3, 224, 224).to(device)
    model = VGG(class_numbers=4, version=16).to(device)
    out = model(input_example)
    print(out.shape)
    print("=== VGG19 ===")
    input_example = torch.rand(4, 3, 224, 224).to(device)
    model = VGG(class_numbers=4, version=19).to(device)
    out = model(input_example)
    print(out.shape)


if __name__ == '__main__':
    verify_model()
