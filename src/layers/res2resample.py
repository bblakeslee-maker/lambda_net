import torch
import torch.nn as nn
from layers import upsampler


class Res2Resample(nn.Module):

    def __init__(self, inputChannels, outputChannels, scales=4,
                 normalization=None, sampleDirection=None,
                 residualDownsample=None, forwardDownsample=None):
        super(Res2Resample, self).__init__()
        if inputChannels % scales != 0:
            raise ValueError('Scales must equally divide output channels!')
        if (residualDownsample == 'pool') or (forwardDownsample == 'pool'):
            if inputChannels != outputChannels:
                raise ValueError('Pooling does not support channel reduction!')
        '''
        if sampleDirection is None and (inputChannels != outputChannels):
            raise ValueError('Same dims does not support channel reduction!')
        '''
        self.scales = scales
        normalization = nn.BatchNorm2d if normalization is None \
            else normalization
        # First level convolution before chunking
        self.convTop = nn.Conv2d(inputChannels, inputChannels,
                                 kernel_size=(1, 1), stride=(1, 1))
        self.normTop = normalization(inputChannels)
        # Multiscale convolution
        self.multiConv = nn.ModuleList(
            [nn.Conv2d(int(inputChannels / scales),
                       int(inputChannels / scales), kernel_size=(3, 3),
                       stride=(1, 1), padding=(1, 1))
             if i != 0 else None for i in range(scales)])
        self.multiNorm = nn.ModuleList(
            [normalization(int(inputChannels / scales))
             if i != 0 else None for i in range(scales)])
        # Residual downsampling function
        if sampleDirection == 'down':
            self.forwardPath, self.bottomLayer = self.__configureDownsampling(
                inputChannels, outputChannels, residualDownsample,
                forwardDownsample)
        elif sampleDirection == 'up':
            # Upsampling function
            self.forwardPath = upsampler.UpsampleConvLayer(
                inputChannels, outputChannels, kernel_size=3, stride=1,
                upsample=2)
            self.bottomLayer = upsampler.UpsampleConvLayer(
                inputChannels, outputChannels, kernel_size=3, stride=1,
                upsample=2)
        else:
            self.bottomLayer = nn.Conv2d(inputChannels, outputChannels,
                                         kernel_size=(1, 1), stride=(1, 1))
            if inputChannels == outputChannels:
                self.forwardPath = nn.Identity()
            else:
                self.forwardPath = nn.Conv2d(inputChannels, outputChannels,
                                             kernel_size=(1, 1), stride=(1, 1))
        self.normBottom = normalization(outputChannels)
        # Miscellaneous layers
        self.relu = nn.ReLU(inplace=True)

    def __configureDownsampling(self, inChannel, outChannel,
                                residualDownsample, forwardDownsample):
        # Residual path downsampling function
        if residualDownsample == 'pool':
            bottomLayer = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        elif residualDownsample == 'conv':
            bottomLayer = nn.Conv2d(inChannel, outChannel, kernel_size=(3, 3),
                                    stride=(2, 2), padding=(1, 1))
        else:
            raise ValueError('Invalid residual downsampling function!')
        # Forward path downsampling function
        if forwardDownsample == 'pool':
            forwardPath = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        elif forwardDownsample == 'conv':
            forwardPath = nn.Conv2d(inChannel, outChannel, kernel_size=(3, 3),
                                    stride=(2, 2), padding=(1, 1))
        else:
            raise ValueError('Invalid forward downsampling function!')
        return forwardPath, bottomLayer

    def forward(self, x):
        # Top layers of Res2Net
        residualOut = self.convTop(x)
        residualOut = self.normTop(residualOut)
        residualOut = self.relu(residualOut)

        # Perform multiscale convolution
        xChunk = torch.chunk(residualOut, self.scales, 1)
        yChunk = list()
        for i in range(self.scales):
            if i == 0:
                yChunk.append(xChunk[i])
            elif i == 1:
                yChunk.append(self.relu(self.multiNorm[i](
                    self.multiConv[i](xChunk[i]))))
            else:
                yChunk.append(self.relu(self.multiNorm[i](
                    self.multiConv[i](xChunk[i] + yChunk[i - 1]))))
        residualOut = torch.cat(yChunk, 1)

        # Perform bottom layer operation
        residualOut = self.bottomLayer(residualOut)
        residualOut = self.normBottom(residualOut)

        # Perform forward path operation
        forwardOut = self.forwardPath(x)
        residualOut = self.relu(residualOut + forwardOut)
        return residualOut
