import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        return self.leaky_relu(self.batch_norm(self.conv(x)))
