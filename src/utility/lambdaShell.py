import torch
import torch.nn as nn
import importlib


class LambdaShell(nn.Module):
    def __init__(self, config):
        super(LambdaShell, self).__init__()
        self.__fusionType = config['arch']['fusionType']
        encoderModel = importlib.import_module(
            'encoders.' + config['arch']['encoderModel'])
        decoderModel = importlib.import_module(
            'decoders.' + config['arch']['decoderModel'])
        self.encoder = encoderModel.Encoder(config)
        self.decoder = decoderModel.Decoder(config)

    def forward(self, refImg, tgtImg):
        fusionOut = dict()
        refOut = self.encoder(refImg)
        tgtOut = self.encoder(tgtImg)
        if self.__fusionType == 'cat':
            for k in refOut:
                fusionOut[k] = torch.cat([refOut[k], tgtOut[k]], dim=1)
        elif self.__fusionType == 'diff':
            for k in refOut:
                fusionOut[k] = tgtOut[k] - refOut[k]
        transformOut = self.decoder(fusionOut)
        return transformOut
