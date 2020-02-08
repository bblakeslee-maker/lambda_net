import torch
from torchvision import models


class Encoder(torch.nn.Module):
    def __init__(self, config, requires_grad=True):
        super(Encoder, self).__init__()
        vggFeatures = models.vgg16(pretrained=False).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for i in range(4):
            self.slice1.add_module(str(i), vggFeatures[i])
        for i in range(4, 9):
            self.slice2.add_module(str(i), vggFeatures[i])
        for i in range(9, 16):
            self.slice3.add_module(str(i), vggFeatures[i])
        for i in range(16, 23):
            self.slice4.add_module(str(i), vggFeatures[i])
        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, x):
        hRelu1_2 = self.slice1(x)
        hRelu2_2 = self.slice2(hRelu1_2)
        hRelu3_3 = self.slice3(hRelu2_2)
        hRelu4_3 = self.slice4(hRelu3_3)
        return {'block4Out': hRelu4_3}
