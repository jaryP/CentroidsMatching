from typing import Union, Tuple

import numpy as np
import torch
import torchvision
from avalanche.models import MultiHeadClassifier, IncrementalClassifier
from torch import nn
from torchvision.models import vgg11

from models import resnet20, resnet32, custom_vgg
from models.utils import CombinedModel, CustomMultiHeadClassifier


def get_backbone(name: str, channels: int = 3):
    name = name.lower()

    if 'vgg' in name:

        model = getattr(torchvision.models, name, None)
        if model is None:
            model = custom_vgg(name)
        else:
            model = model()

        # if name == 'vgg11':
        #     model = vgg11()

        feat = model.features
        model = nn.Sequential(feat, nn.AdaptiveAvgPool2d((7, 7)))

        return model
    elif name == 'alexnet':
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.25),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.25),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.25),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.25),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            # nn.Dropout2d(0.25),
        )
    elif 'resnet' in name:
        if name == 'resnet20':
            return resnet20()
        elif name == 'resnet32':
            return resnet32()

    assert False, f'Inserted model name not valid {name}'


def get_cl_model(model_name: str,
                 method_name: str,
                 input_shape: Tuple[int, int, int],
                 sit: bool = False,
                 cml_out_features: int = 256):

    backbone = get_backbone(model_name, channels=input_shape[0])
    x = torch.randn((1,) + input_shape)
    o = backbone(x)

    size = np.prod(o.shape)

    def heads_generator(i, o):
        return nn.Sequential(nn.ReLU(),
                             nn.Linear(i, i),
                             nn.ReLU(),
                             nn.Linear(i, o))

    if method_name != 'cml':
        if sit:
            classifier = IncrementalClassifier(size)
        else:
            # classifier = CustomMultiHeadClassifier(size, heads_generator)
            classifier = MultiHeadClassifier(size)
    else:
        if cml_out_features is None:
            cml_out_features = size

        classifier = CustomMultiHeadClassifier(size, heads_generator,
                                               cml_out_features)

    model = CombinedModel(backbone, classifier)

    return model
