import torch
import torch.nn as nn

from models.utils import ConvBlock

RESNET_VERSIONS = {
    18: {
        'block_type': 'basic',
        'stages': [2, 2, 2, 2]  # elements are block numbers before downsampling
    },
    34: {
        'block_type': 'basic',
        'stages': [3, 4, 6, 3]
    },
    50: {
        'block_type': 'bottleneck',
        'stages': [3, 4, 6, 3]
    },
    101: {
        'block_type': 'bottleneck',
        'stages': [3, 4, 23, 3]
    },
    152: {
        'block_type': 'bottleneck',
        'stages': [3, 8, 36, 3]
    },
}


class BasicResNetBlock(nn.Module):
    def __init__(self, channels, halve_the_size=False):
        super().__init__()

        self.halve_the_size = halve_the_size
        first_stride = 2 if halve_the_size else 1
        first_channel = channels // 2 if halve_the_size else channels

        self.first_conv = nn.Conv2d(first_channel, channels, kernel_size=(3, 3), stride=(first_stride, first_stride),
                                    padding=(1, 1), bias=False)
        self.second_conv = nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        self.first_batch_norm = nn.BatchNorm2d(channels)
        self.second_batch_norm = nn.BatchNorm2d(channels)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

        if self.halve_the_size:
            self.downsample_x = nn.Sequential(
                nn.Conv2d(channels // 2, channels, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0), bias=False),
                nn.BatchNorm2d(channels)
            )

    def forward(self, x):
        x_cloned = x.clone()
        if self.halve_the_size:
            x_cloned = self.downsample_x(x_cloned)

        x = self.first_conv(x)
        x = self.first_batch_norm(x)
        x = self.leaky_relu(x)
        x = self.second_conv(x)
        x = self.second_batch_norm(x)

        x += x_cloned

        return self.leaky_relu(x)


class BottleneckResidualBlock(nn.Module):
    def __init__(self, channels, halve_the_size=False):
        super().__init__()

        self.halve_the_size = halve_the_size
        second_stride = 2 if halve_the_size else 1
        first_channel = channels // 2 if halve_the_size else channels

        self.first_conv = nn.Conv2d(first_channel, channels // 4, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                    bias=False)
        self.second_conv = nn.Conv2d(channels // 4, channels // 4, kernel_size=(3, 3),
                                     stride=(second_stride, second_stride), padding=(1, 1), bias=False)
        self.third_conv = nn.Conv2d(channels // 4, channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                    bias=False)

        self.first_batch_norm = nn.BatchNorm2d(channels // 4)
        self.second_batch_norm = nn.BatchNorm2d(channels // 4)
        self.third_batch_norm = nn.BatchNorm2d(channels)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

        if self.halve_the_size:
            self.downsample_x = nn.Sequential(
                nn.Conv2d(channels // 2, channels, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0), bias=False),
                nn.BatchNorm2d(channels)
            )

    def forward(self, x):
        x_cloned = x.clone()
        if self.halve_the_size:
            x_cloned = self.downsample_x(x_cloned)

        x = self.first_conv(x)
        x = self.first_batch_norm(x)
        x = self.leaky_relu(x)
        x = self.second_conv(x)
        x = self.second_batch_norm(x)
        x = self.leaky_relu(x)
        x = self.third_conv(x)
        x = self.third_batch_norm(x)

        x += x_cloned

        return self.leaky_relu(x)


class ResNet(nn.Module):
    def __init__(self, version, class_numbers, in_channels=3):
        super().__init__()

        assert version in RESNET_VERSIONS

        self.in_channels = in_channels
        self.version = version

        self.stem_layers = self._create_stem_layers()
        self.resnet_blocks = self._create_resnet_blocks()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Linear(512, class_numbers)

    def forward(self, x):
        x = self.stem_layers(x)
        x = self.resnet_blocks(x)
        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)
        return self.linear(x)

    def _create_stem_layers(self):
        return nn.Sequential(
            ConvBlock(in_channels=self.in_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def _create_resnet_blocks(self):
        layers = []
        Block = BasicResNetBlock if RESNET_VERSIONS[self.version]['block_type'] == 'basic' else BottleneckResidualBlock

        channels = 64
        stages = RESNET_VERSIONS[self.version]['stages']
        for stage_idx, block_numbers in enumerate(stages):
            for block_idx in range(block_numbers):
                if block_idx == 0 and stage_idx != 0:
                    layers += [Block(channels, halve_the_size=True)]
                else:
                    layers += [Block(channels, halve_the_size=False)]
            channels *= 2

        return nn.Sequential(*layers)


def verify_resnet_block():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_example = torch.rand(2, 5, 40, 40).to(device)

    """basic block"""
    block1 = BasicResNetBlock(channels=5, halve_the_size=False).to(device)
    block2 = BasicResNetBlock(channels=5, halve_the_size=True).to(device)

    out = block1(input_example)
    assert list(out.shape) == [2, 5, 40, 40]
    out = block2(input_example)
    assert list(out.shape) == [2, 5, 20, 20]

    """Bottleneck block"""
    block1 = BottleneckResidualBlock(channels=5, halve_the_size=False).to(device)
    block2 = BottleneckResidualBlock(channels=5, halve_the_size=True).to(device)

    out = block1(input_example)
    assert list(out.shape) == [2, 5, 40, 40]
    out = block2(input_example)
    assert list(out.shape) == [2, 5, 20, 20]


def verify_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_example = torch.rand(2, 3, 224, 224).to(device)
    model = ResNet(version=18, num_classes=4).to(device)
    out = model(input_example)
    print(out.shape)


if __name__ == '__main__':
    verify_resnet_block()
    verify_model()
