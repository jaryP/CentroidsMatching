from collections import defaultdict

import torch
from torch import nn

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512,
          'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
          512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512,
          512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, cfg, init_weights=True):
        super(VGG, self).__init__()

        layers = nn.ModuleList()
        in_channels = 3

        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

        self.layers = layers

        self.branches = sum([1 for v in cfg if isinstance(v, int)])

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        intermediate = []
        for m in self.layers:
            x = m(x)
            if isinstance(m, nn.Conv2d):
                intermediate.append(x)

        intermediate = intermediate[:-1]

        x = self.avgpool(x)
        intermediate.append(x)

        return intermediate

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def vgg11(**kwargs):
    return VGG(cfgs['A'])

