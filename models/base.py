from typing import Tuple

import numpy as np
import torch
import torchvision
from avalanche.models import MultiHeadClassifier, IncrementalClassifier
from torch import nn
import torch.nn.functional as F

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

        feat = model.features
        model = nn.Sequential(feat, nn.AdaptiveAvgPool2d((7, 7)))

        return model

    elif name == 'alexnet':
        s = nn.Sequential(
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

        class Wrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.model = m

            def forward(self, x, task_label=None, **kwargs):
                return self.model(x)

        return Wrapper(s)

    elif 'resnet' in name:
        if name == 'resnet20':
            model = resnet20()
        elif name == 'resnet32':
            model = resnet32()
        else:
            assert False

        class CustomResNet(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.model = m

            def forward(self, x, task_label=None, **kwargs):
                out = F.relu(self.model.bn1(self.model.conv1(x)))
                out = self.model.layer1(out)
                out = self.model.layer2(out)
                out = self.model.layer3(out)
                out = F.avg_pool2d(out, out.size()[3])
                out = out.view(out.size(0), -1)
                return out

        return CustomResNet(model)

    assert False, f'Inserted model name not valid {name}'


def get_cl_model(model_name: str,
                 method_name: str,
                 input_shape: Tuple[int, int, int],
                 sit: bool = False,
                 cml_out_features: int = None,
                 is_stream: bool = False):
    backbone = get_backbone(model_name, channels=input_shape[0])

    if method_name in ['cope', 'mcml']:
        return backbone

    x = torch.randn((1,) + input_shape)
    o = backbone(x)

    size = np.prod(o.shape)

    def heads_generator(i, o):
        class Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Sequential(
                    # nn.Dropout(0.5),
                    nn.Linear(i, i),
                    nn.ReLU(),
                    nn.Linear(i, o),
                    # nn.Dropout(0.2)
                )

            def forward(self, x, task_labels=None, **kwargs):
                return self.model(x)

        return Wrapper()

    if is_stream or method_name == 'icarl':
        classifier = IncrementalClassifier(size)
    else:
        if method_name == 'er':
            classifier = CustomMultiHeadClassifier(size, heads_generator)
        else:
            if method_name != 'cml':
                if sit:
                    classifier = IncrementalClassifier(size)
                else:
                    classifier = MultiHeadClassifier(size)
            else:
                if cml_out_features is None:
                    cml_out_features = 128

                classifier = CustomMultiHeadClassifier(size, heads_generator,
                                                       cml_out_features)

    model = CombinedModel(backbone, classifier)

    return model
