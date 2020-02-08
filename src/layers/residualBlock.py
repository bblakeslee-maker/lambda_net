import torch.nn as nn
from layers import convLayer


class ResidualBlock(nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = convLayer.ConvLayer(channels, channels, kernel_size=3,
                                         stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = convLayer.ConvLayer(channels, channels, kernel_size=3,
                                         stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out
