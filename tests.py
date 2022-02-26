from itertools import chain

import numpy as np
import torch
from avalanche.benchmarks import SplitCIFAR10
from avalanche.models import as_multitask
from avalanche.training import BaseStrategy, GEM
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, \
    RandomCrop, Normalize

from avalanche.evaluation.metrics import forgetting_metrics, \
    accuracy_metrics, loss_metrics, timing_metrics, cpu_usage_metrics, \
    confusion_matrix_metrics, disk_usage_metrics, bwt_metrics, BWT

from methods.strategies import EmbeddingRegularization
# from models.base import MultiHeadBackbone, EmbeddingModelDecorator, \
#     CustomMultiTaskDecorator
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin

from models.base import get_cl_model
from models.utils import CustomMultiTaskDecorator


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # self.classifier = nn.Sequential(
        self.features = nn.Sequential(
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

        self.classifier = nn.Sequential(nn.Linear(256, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 1))

        #     nn.Dropout(),
        #     nn.Linear(256 * 2 * 2, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, num_classes),
        # )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        # x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x


class Head(nn.Module):
    def __init__(self, input_size, classes):
        super().__init__()

        if isinstance(input_size, (tuple, list)):
            input_size = np.mul(input_size)

        self.model = nn.Linear(input_size, classes)

    def forward(self, x):
        return self.model(x)


backbone = AlexNet()

# model = MultiHeadBackbone(backbone=backbone,
#                           backbone_output_size=256,
#                           incremental_classifier_f=lambda x, y: Head(x, y))

# model = get_cl_model('resnet20', 'gem', (3, 32, 32))

train_transform = Compose([ToTensor(),
                           RandomCrop(32, padding=4),
                           RandomHorizontalFlip(),
                           Normalize(
                               [0.4914, 0.4822, 0.4465],
                               (0.2023, 0.1994, 0.2010))
                           ])

test_transform = Compose([ToTensor(),
                          Normalize(
                              (0.4914, 0.4822, 0.4465),
                              (0.2023, 0.1994, 0.2010))
                          ])

tasks = SplitCIFAR10(n_experiences=5,
                     return_task_id=True,
                     train_transform=train_transform,
                     eval_transform=test_transform,
                     )

# model = as_multitask(backbone, 'classifier')
#
# model = EmbeddingModelDecorator(model)

def heads_generator(i, o):
    return nn.Sequential(nn.Linear(i, i // 2),
                         nn.ReLU(),
                         nn.Linear(i // 2, o))

model = CustomMultiTaskDecorator(backbone, 'classifier', heads_generator)
# model = EmbeddingModelDecorator(model)

parameters = chain(model.parameters())

opt = Adam(parameters, lr=0.001)

criterion = CrossEntropyLoss()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=False, epoch=False, experience=True,
                     stream=True),
    # loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    # timing_metrics(epoch=True),
    # forgetting_metrics(experience=True, stream=True),
    # cpu_usage_metrics(experience=True),
    # confusion_matrix_metrics(num_classes=tasks.n_classes, save_image=False, stream=True),
    # disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    bwt_metrics(experience=True, stream=True),
    loggers=[InteractiveLogger()],
    benchmark=tasks,
    strict_checks=True
)

strategy = BaseStrategy(model=model,
                        criterion=criterion,
                        optimizer=opt,
                        train_epochs=10,
                        train_mb_size=32,
                        evaluator=eval_plugin,
                        device='cuda:0')

# strategy = EmbeddingRegularization(model=model,
#                                    penalty_weight=10,
#                                    mem_size=500,
#                                    criterion=criterion,
#                                    optimizer=opt,
#                                    train_epochs=5,
#                                    train_mb_size=32,
#                                    evaluator=eval_plugin,
#                                    device='cuda:0')

strategy = GEM(model=model,
               patterns_per_exp=200,
               criterion=criterion,
               optimizer=opt,
               train_epochs=5,
               train_mb_size=32,
               evaluator=eval_plugin,
               device='cuda:0')

results = []
for experience in tasks.train_stream:
    print('task')
    strategy.train(experiences=experience)

    results.append(strategy.eval(tasks.test_stream))
    # print(model)

for k, v in eval_plugin.get_last_metrics().items():
    print(k, v)

# print(eval_plugin.get_last_metrics())

# print(strategy.eval(tasks.test_stream))
# print(results[-1])
print(model)
