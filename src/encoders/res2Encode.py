import torch.nn as nn
from layers import res2resample


class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder, self).__init__()
        # Configuration flags
        mode = config['arch']['mode']
        if mode == 'lambda' or mode == 'omega':
            self.mode = mode
        else:
            raise ValueError(
                'Mode {} is not a valid setting. '
                'Use either lambda or omega.'.format(mode))
        # Non-learning layers
        self.relu = nn.ReLU(inplace=True)
        # Block 1
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1),
                                 padding=(1, 1))
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1),
                                 padding=(1, 1))
        # Block 2
        self.pool2_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0,
                                    dilation=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1),
                                 padding=(1, 1))
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1),
                                 padding=(1, 1))
        # Block 3
        self.pool3_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0,
                                    dilation=1, ceil_mode=False)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1),
                                 padding=(1, 1))
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1),
                                 padding=(1, 1))
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1),
                                 padding=(1, 1))
        # Block 4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1),
                                 padding=(1, 1))
        self.res4_1 = res2resample.Res2Resample(
            512, 512, scales=4, sampleDirection='down',
            residualDownsample='pool', forwardDownsample='pool')

    def forward(self, x):
        # Block 1
        y = self.conv1_1(x)
        y = self.relu(y)
        y = self.conv1_2(y)
        block1Out = self.relu(y)
        # Block 2
        y = self.pool2_1(block1Out)
        y = self.conv2_1(y)
        y = self.relu(y)
        y = self.conv2_2(y)
        block2Out = self.relu(y)
        # Block 3
        y = self.pool3_1(block2Out)
        y = self.conv3_1(y)
        y = self.relu(y)
        y = self.conv3_2(y)
        y = self.relu(y)
        y = self.conv3_3(y)
        block3Out = self.relu(y)
        # Block 4
        y = self.conv4_1(block3Out)
        y = self.relu(y)
        block4Out = self.res4_1(y)  # Contains integrated ReLU
        if self.mode == 'lambda':
            return {'block4Out': block4Out}
        elif self.mode == 'omega':
            return {'block1Out': block1Out, 'block2Out': block2Out,
                    'block3Out': block3Out, 'block4Out': block4Out}
