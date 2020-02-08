import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        fusionType = config['arch']['fusionType']
        mode = config['arch']['mode']
        outputChannels = config['arch'].getint('outputChannels')
        if fusionType == 'cat':
            inputChannels = 1024
        elif fusionType == 'diff':
            inputChannels = 512
        # Block 1
        self.deconv1_1 = nn.ConvTranspose2d(in_channels=inputChannels,
                                            out_channels=512,
                                            kernel_size=(3, 3), stride=(1, 1),
                                            padding=(1, 1))
        self.relu1_1 = nn.ReLU(inplace=True)
        self.deconv1_2 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                            kernel_size=(3, 3), stride=(1, 1),
                                            padding=(1, 1))
        self.relu1_2 = nn.ReLU(inplace=True)
        self.deconv1_3 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=(3, 3),
            stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
        self.relu1_3 = nn.ReLU(inplace=True)
        # Block 2
        self.deconv2_1 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                            kernel_size=(3, 3), stride=(1, 1),
                                            padding=(1, 1))
        self.relu2_1 = nn.ReLU(inplace=True)
        self.deconv2_2 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                            kernel_size=(3, 3), stride=(1, 1),
                                            padding=(1, 1))
        self.relu2_2 = nn.ReLU(inplace=True)
        self.deconv2_3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=(3, 3),
            stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
        self.relu2_3 = nn.ReLU(inplace=True)
        # Block 3
        self.deconv3_1 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                            kernel_size=(3, 3), stride=(1, 1),
                                            padding=(1, 1))
        self.relu3_1 = nn.ReLU(inplace=True)
        self.deconv3_2 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=(3, 3),
            stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
        self.relu3_2 = nn.ReLU(inplace=True)
        # Block 4
        self.deconv4_1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                            kernel_size=(3, 3), stride=(1, 1),
                                            padding=(1, 1))
        self.relu4_1 = nn.ReLU(inplace=True)
        self.deconv4_2 = nn.ConvTranspose2d(in_channels=64,
                                            out_channels=outputChannels,
                                            kernel_size=(3, 3), stride=(1, 1),
                                            padding=(1, 1), dilation=1)

    def forward(self, x):
        out = self.deconv1_1(x['block4Out'])
        out = self.relu1_1(out)
        out = self.deconv1_2(out)
        out = self.relu1_2(out)
        out = self.deconv1_3(out)
        out = self.relu1_3(out)
        out = self.deconv2_1(out)
        out = self.relu2_1(out)
        out = self.deconv2_2(out)
        out = self.relu2_2(out)
        out = self.deconv2_3(out)
        out = self.relu2_3(out)
        out = self.deconv3_1(out)
        out = self.relu3_1(out)
        out = self.deconv3_2(out)
        out = self.relu3_2(out)
        out = self.deconv4_1(out)
        out = self.relu4_1(out)
        out = self.deconv4_2(out)
        return out
