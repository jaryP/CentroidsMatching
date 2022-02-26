from collections import defaultdict

import torch
from torch import nn
from torchvision.models import VGG
from torchvision.models.vgg import make_layers

cfgs = {
    'half_vgg11': [32, 'M', 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256, 'M'],
}


def custom_vgg(name, batch_norm = False, **kwargs):
    model = VGG(make_layers(cfgs[name], batch_norm=batch_norm), **kwargs)
    return model