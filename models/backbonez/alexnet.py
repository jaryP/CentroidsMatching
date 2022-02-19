from collections import defaultdict

import torch
from torch import nn


class AlexNet(nn.Module):

    def __init__(self, channels=3):
        super(AlexNet, self).__init__()

        self.b = 5
        self.c1 = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=2)
        self.c2 = nn.Conv2d(64, 192, kernel_size=3, padding=2)
        self.c3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.c4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.c5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

    def forward(self, x):
        intermediate_layers = []

        for i in range(1, 6):
            cl = getattr(self, 'c{}'.format(i))
            x = cl(x)

            if i in [2, 3, 5]:
                x = nn.functional.max_pool2d(x, kernel_size=2)

            x = torch.relu(x)
            intermediate_layers.append(x)

        # intermediate_layers.append(x)
        # input(len(intermediate_layers))

        # x = self.c1(x)
        # intermediate_layers.append(x)
        # x = nn.functional.max_pool2d(x, kernel_size=2)
        # x = torch.relu(x)
        #
        # x = self.c2(x)
        # intermediate_layers.append(x)
        # x = nn.functional.max_pool2d(x, kernel_size=2)
        # x = torch.relu(x)
        #
        # x = self.c3(x)
        # intermediate_layers.append(x)
        # x = torch.relu(x)
        #
        # x = self.c4(x)
        # intermediate_layers.append(x)
        # x = torch.relu(x)
        #
        # # self.c5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # x = self.c5(x)
        # intermediate_layers.append(x)
        # x = nn.functional.max_pool2d(x, kernel_size=3, stride=2)
        # x = torch.relu(x)
        # # conv_features = self.features(x)

        # flatten = torch.flatten(x, 1)
        # fc = self.fc_layers(flatten)

        return intermediate_layers


class AlexnetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.fc_layers = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        return self.fc_layers(torch.flatten(x, 1))
