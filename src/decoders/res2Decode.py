import torch
import torch.nn as nn
from layers import res2resample


class Decoder(nn.Module):

    def __init__(self, config):
        super(Decoder, self).__init__()
        # Configuration flags
        mode = config['arch']['mode']
        if mode == 'lambda' or mode == 'omega':
            self.mode = mode
            if mode == 'lambda':
                self.channelScaling = 1
            elif mode == 'omega':
                self.channelScaling = 3
        else:
            raise ValueError(
                'Mode {} is not a valid setting. '
                'Use either lambda or omega.'.format(mode))
        fusionType = config['arch']['fusionType']
        if fusionType == 'cat':
            inputChannels = 1024
        elif fusionType == 'diff':
            inputChannels = 512
        else:
            raise ValueError('Fusion type {} is not a valid setting.'
                             'Use either cat or diff.'.format(fusionType))
        outputChannels = config['arch'].getint('outputChannels')
        # Non-learning layers
        self.relu = nn.ReLU(inplace=True)
        # Block 4
        self.res4_1 = res2resample.Res2Resample(inputChannels, 512, scales=4)
        self.deconv4_1 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=(3, 3),
            stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
        # Block 3
        self.deconv3_1 = nn.ConvTranspose2d(
            in_channels=256 * self.channelScaling, out_channels=256,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.deconv3_2 = nn.ConvTranspose2d(
            in_channels=256, out_channels=256, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1))
        self.deconv3_3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=(3, 3),
            stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
        # Block 2
        self.deconv2_1 = nn.ConvTranspose2d(
            in_channels=128 * self.channelScaling, out_channels=128,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.deconv2_2 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=(3, 3),
            stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
        # Block 1
        self.deconv1_1 = nn.ConvTranspose2d(
            in_channels=64 * self.channelScaling, out_channels=64,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.deconv1_2 = nn.ConvTranspose2d(
            in_channels=64, out_channels=outputChannels, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1), dilation=1)

    def forward(self, x):
        # Block 4
        y = self.res4_1(x['block4Out'])
        y = self.deconv4_1(y)
        block4Out = self.relu(y)
        # Block 3
        if self.mode == 'omega':
            block4Out = torch.cat([block4Out, x['block3Out']], dim=1)
        y = self.deconv3_1(block4Out)
        y = self.relu(y)
        y = self.deconv3_2(y)
        y = self.relu(y)
        y = self.deconv3_3(y)
        block3Out = self.relu(y)
        # Block 2
        if self.mode == 'omega':
            block3Out = torch.cat([block3Out, x['block2Out']], dim=1)
        y = self.deconv2_1(block3Out)
        y = self.relu(y)
        y = self.deconv2_2(y)
        block2Out = self.relu(y)
        # Block 1
        if self.mode == 'omega':
            block2Out = torch.cat([block2Out, x['block1Out']], dim=1)
        y = self.deconv1_1(block2Out)
        y = self.relu(y)
        block1Out = self.deconv1_2(y)
        return block1Out
