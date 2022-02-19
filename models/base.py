from typing import Union, Tuple

import numpy as np
import torch
import torchvision
from avalanche.models import MultiHeadClassifier, IncrementalClassifier
from torch import nn
from torchvision.models import vgg11

from models import resnet20, resnet32
from models.utils import CombinedModel, CustomMultiHeadClassifier


def get_backbone(name: str, channels: int = 3):
    name = name.lower()

    if 'vgg' in name:

        model = getattr(torchvision.models, name)()
        # if name == 'vgg11':
        #     model = vgg11()

        feat = model.features
        model = nn.Sequential(feat, nn.AdaptiveAvgPool2d((7, 7)))

        return model


    elif 'resnet' in name:
        if name == 'resnet20':
            return resnet20()
        elif name == 'resnet32':
            return resnet32()

    assert False, f'Inserted model name not valid {name}'


def get_cl_model(model_name: str,
                 method_name: str,
                 input_shape: Tuple[int, int, int],
                 sit: bool = False):
    backbone = get_backbone(model_name, channels=input_shape[0])
    x = torch.randn((1,) + input_shape)
    o = backbone(x)

    size = np.prod(o.shape)

    if method_name != 'clm':
        if sit:
            classifier = MultiHeadClassifier(size)
        else:
            classifier = IncrementalClassifier(size)

    else:
        def heads_generator(i, o):
            return nn.Sequential(nn.ReLU(),
                                 nn.Linear(i, i),
                                 nn.ReLU(),
                                 nn.Linear(i, o))

        classifier = CustomMultiHeadClassifier(size, heads_generator,
                                               size)

    model = CombinedModel(backbone, classifier)

    return model
