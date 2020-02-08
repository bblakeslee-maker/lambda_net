import torch.nn as nn
from layers import convLayer, residualBlock, upsampler


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        if config['arch']['fusionType'] == 'cat':
            inputChannels = 1024
        elif config['arch']['fusionType'] == 'diff':
            inputChannels = 512
        self.res1 = residualBlock.ResidualBlock(inputChannels)
        self.res2 = residualBlock.ResidualBlock(inputChannels)
        self.res3 = residualBlock.ResidualBlock(inputChannels)
        # Upsampling Layers
        self.deconv1 = upsampler.UpsampleConvLayer(inputChannels, 512,
                                                   kernel_size=3,
                                                   stride=1, upsample=2)
        if config['arch']['normType'] == 'inst':
            print('STATUS: Enabling in4 instance norm!')
            self.in4 = nn.InstanceNorm2d(512, affine=True)
        elif config['arch']['normType'] == 'batch':
            print('STATUS: Enabling in4 batch norm!')
            self.in4 = nn.BatchNorm2d(512, affine=True)
        else:
            print('STATUS: Disabling in4 normalization!')
            self.in4 = nn.Identity()
        self.deconv2 = upsampler.UpsampleConvLayer(512, 256, kernel_size=3,
                                                   stride=1, upsample=2)
        if config['arch']['normType'] == 'inst':
            print('STATUS: Enabling in5 instance norm!')
            self.in5 = nn.InstanceNorm2d(256, affine=True)
        elif config['arch']['normType'] == 'batch':
            print('STATUS: Enabling in5 batch norm!')
            self.in5 = nn.BatchNorm2d(256, affine=True)
        else:
            print('STATUS: Disabling in5 normalization!')
            self.in5 = nn.Identity()
        self.deconv3 = upsampler.UpsampleConvLayer(256, 128, kernel_size=3,
                                                   stride=1, upsample=2)
        if config['arch']['normType'] == 'inst':
            print('STATUS: Enabling in6 instance norm!')
            self.in6 = nn.InstanceNorm2d(128, affine=True)
        elif config['arch']['normType'] == 'batch':
            print('STATUS: Enabling in6 batch norm!')
            self.in6 = nn.BatchNorm2d(128, affine=True)
        else:
            print('STATUS: Disabling in6 normalization!')
            self.in6 = nn.Identity()
        self.deconv4 = convLayer.ConvLayer(
            128, config['arch'].getint('outputChannels'),
            kernel_size=9, stride=1)
        # Non-linearities
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.res1(x['block4Out'])
        y = self.res2(y)
        y = self.res3(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.relu(self.in6(self.deconv3(y)))
        y = self.deconv4(y)
        return y
