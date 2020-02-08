import torch.nn as nn
from layers import convLayer, residualBlock


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        # Initial convolution layers
        self.conv1 = convLayer.ConvLayer(3, 32, kernel_size=9, stride=1)
        if config['arch']['normType'] == 'inst':
            print('STATUS: Enabling in1 instance norm!')
            self.in1 = nn.InstanceNorm2d(32, affine=True)
        elif config['arch']['normType'] == 'batch':
            print('STATUS: Enabling in1 batch norm!')
            self.in1 = nn.BatchNorm2d(32, affine=True)
        else:
            print('STATUS: Disabling in1 normalization!')
            self.in1 = nn.Identity()
        self.conv2 = convLayer.ConvLayer(32, 64, kernel_size=3, stride=2)
        if config['arch']['normType'] == 'inst':
            print('STATUS: Enabling in2 instance norm!')
            self.in2 = nn.InstanceNorm2d(64, affine=True)
        elif config['arch']['normType'] == 'batch':
            print('STATUS: Enabling in2 batch norm!')
            self.in2 = nn.BatchNorm2d(64, affine=True)
        else:
            print('STATUS: Disabling in2 normalization!')
            self.in2 = nn.Identity()
        self.conv3 = convLayer.ConvLayer(64, 128, kernel_size=3, stride=2)
        if config['arch']['normType'] == 'inst':
            print('STATUS: Enabling in3 instance norm!')
            self.in3 = nn.InstanceNorm2d(128, affine=True)
        elif config['arch']['normType'] == 'batch':
            print('STATUS: Enabling in3 batch norm!')
            self.in3 = nn.BatchNorm2d(128, affine=True)
        else:
            print('STATUS: Disabling in3 instance norm!')
            self.in3 = nn.Identity()
        # Residual layers
        self.res1 = residualBlock.ResidualBlock(128)
        self.res2 = residualBlock.ResidualBlock(128)
        self.res3 = residualBlock.ResidualBlock(128)
        # Non-linearities
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.relu(self.in1(self.conv1(x)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        return {'block4Out': y}
