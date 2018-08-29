import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import math
from torch.autograd import Variable
import torch.nn.init as init


class VGG19(nn.Module):

    def __init__(self, cfg=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M', 1024, 1024, 1024, 1024, 'M']):
        super(VGG19, self).__init__()
        self.features = self._make_layers(cfg)
        self.classifier = nn.Linear(6144, 375)

    def forward(self, x):
        out = self.features(x)
        print(out.size())
        out = out.view(out.size(0), -1)
        print(out.size())
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
