import torch
import torch.nn as nn


class BasicResNetBlock(nn.Module):
    def __init__(self, channels, halve_the_size=False):
        super().__init__()

        self.halve_the_size = halve_the_size
        first_stride = 2 if halve_the_size else 1

        self.first_conv = nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(first_stride, first_stride),
                                    padding=(1, 1), bias=False)
        self.second_conv = nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        self.first_batch_norm = nn.BatchNorm2d(channels)
        self.second_batch_norm = nn.BatchNorm2d(channels)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

        if self.halve_the_size:
            self.downsample_x = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0), bias=False),
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

        self.first_conv = nn.Conv2d(channels, channels // 4, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
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
                nn.Conv2d(channels, channels, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0), bias=False),
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
    pass


if __name__ == '__main__':
    verify_resnet_block()
    verify_model()
