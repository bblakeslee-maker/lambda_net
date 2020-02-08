import torch
from torchvision import models


class Encoder(torch.nn.Module):
    def __init__(self, config, requires_grad=False):
        super(Encoder, self).__init__()
        vggPretrainedFeatures = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for i in range(4):
            self.slice1.add_module(str(i), vggPretrainedFeatures[i])
        for i in range(4, 9):
            self.slice2.add_module(str(i), vggPretrainedFeatures[i])
        for i in range(9, 16):
            self.slice3.add_module(str(i), vggPretrainedFeatures[i])
        for i in range(16, 23):
            self.slice4.add_module(str(i), vggPretrainedFeatures[i])
        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, x):
        hRelu1_2 = self.slice1(x)
        hRelu2_2 = self.slice2(hRelu1_2)
        hRelu3_3 = self.slice3(hRelu2_2)
        hRelu4_3 = self.slice4(hRelu3_3)
        return {'block4Out': hRelu4_3}
