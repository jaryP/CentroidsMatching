from collections import defaultdict
from copy import deepcopy
from itertools import chain
from typing import Sequence, List

import numpy as np
import torch
from avalanche.benchmarks import SplitCIFAR10
from avalanche.models import DynamicModule
from torch import nn, log_softmax, softmax, cosine_similarity
from torch.nn import CrossEntropyLoss
from torch.nn.functional import normalize
from torch.optim import Adam
from torch.utils.data import Subset, DataLoader

from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, \
    RandomCrop, Normalize

from avalanche.evaluation.metrics import forgetting_metrics, \
    accuracy_metrics, loss_metrics, timing_metrics, cpu_usage_metrics, \
    confusion_matrix_metrics, disk_usage_metrics, bwt_metrics, BWT
from tqdm import tqdm

from methods.strategies import EmbeddingRegularization, ContinualMetricLearning
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin

from models.utils import CustomMultiTaskDecorator, EmbeddingModelDecorator


def model_adaptation(model, dataset, device):
    """Adapts the model to the current data.

    Calls the :class:`~avalanche.models.DynamicModule`s adaptation.
    """
    for module in model.modules():
        if isinstance(module, DynamicModule):
            module.adaptation(dataset)

    return model.to(device)


def calculate_similarity(x, y, similarity='euclidean', sigma=1):
    # if isinstance(y, list):
    #     if isinstance(y[0], MultivariateNormal):
    #         diff = torch.stack([-likelihhod(n, x) for n in y], 1)
    #         # diff = torch.stack([-n.log_prob(x) ** 10 for n in y], 1)
    #         return diff
    #     elif isinstance(y[0], tuple):
    #         diff = torch.stack([-likelihhod(n, x) for n in y], 1)
    #         return diff

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    a = x.unsqueeze(1).expand(n, m, d)
    b = y.unsqueeze(0).expand(n, m, d)

    if similarity == 'euclidean':
        dist = -torch.pow(a - b, 2).sum(2).sqrt()
    elif similarity == 'rbf':
        dist = -torch.pow(a - b, 2).sum(2).sqrt()
        dist = dist / (2 * sigma ** 2)
        dist = torch.exp(dist)
    elif similarity == 'cosine':
        dist = cosine_similarity(a, b, -1) ** 2
    else:
        assert False

    return dist

def calculate_centroids(data: Sequence,
                        model,
                        ti,
                        gaussian: bool = False):

    device = next(backbone.parameters()).device

    embs = defaultdict(list)

    if isinstance(data, DataLoader):
        classes = set()
        for d in data:
            x, y, _ = d
            classes.update(y.detach().cpu().tolist())

        classes = sorted(classes)

        for d in data:
            x, y, _ = d
            x = x.to(device)
            embeddings = model.forward_single_task(x, ti, False)

            for c in classes:
                embs[c].append(embeddings[y == c])

        embs = {c: torch.cat(e, 0) for c, e in embs.items()}
        # return torch.stack([torch.mean(torch.cat(embs[e], 0), 0)
        #                     for e in sorted(classes)], 0)

    else:

        x, y = data
        classes = sorted(set(y.detach().cpu().tolist()))

        x = x.to(device)

        embeddings = model.forward_single_task(x, ti, False)
        embs = {c: embeddings[y == c]
                for c in classes}

        # return torch.stack(centroids, 0)

    if not gaussian:
        centroids = torch.stack([torch.mean(embs[c], 0) for c in classes], 0)
        return centroids
    else:
        eye = torch.eye(embs[classes[0]].shape[-1], device=device) * 1e-8

        means = torch.stack([torch.mean(embs[c], 0) for c in classes], 0)
        variances = torch.stack([torch.cov(embs[c].T) + eye for c in classes], 0)

        return means, variances

        return [MultivariateNormal(torch.mean(embs[c], 0),
                                   torch.cov(embs[c].T) + eye)
                for c in classes]
        # return [MultivariateNormal(m, s) for m, s in zip(means, variances)]


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

# model = EmbeddingModelDecorator(model)

def heads_generator(i, o):
    return nn.Sequential(nn.ReLU(),
                         nn.Linear(i, i),
                         nn.ReLU(),
                         nn.Linear(i, o))


model = CustomMultiTaskDecorator(backbone, 'classifier', heads_generator,
                                 out_features=256)
model = EmbeddingModelDecorator(model)


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

classes_centroids = []
tasks_centroids = []

device = 'cuda:0'
epochs = 20

train_stream = tasks.streams['train']
test_stream = tasks.streams['test']

tasks_matrix_scores = np.full((len(test_stream),
                               len(test_stream)), -1.0)


for ti, (tr_s, te_s) in enumerate(zip(train_stream, test_stream)):
    # model.adaptation(tr_s.dataset)
    model = model_adaptation(model, tr_s.dataset, device)

    # classes = tr_s.classes_in_this_experience
    # print(classes)
    # for pit in range(ti):
    #
    #     for param in model.wrapped_class.classifier.classifiers[str(pit)].parameters():
    #         param.requires_grad_(False)

    train = tr_s.dataset
    test = te_s.dataset

    idx = np.arange(len(train))
    np.random.shuffle(idx)
    dev_i = int(len(idx) * 0.1)

    dev_idx = idx[:dev_i]
    train_idx = idx[dev_i:]

    dev = Subset(train.eval(), dev_idx)
    train = Subset(train, train_idx)

    if ti > 0:
        past_model = deepcopy(model)

    train_dataloader = DataLoader(train, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=32)
    dev_dataloader = DataLoader(dev, batch_size=len(dev))

    parameters = chain(
        model.parameters(),
    )

    opt = Adam(parameters, lr=0.001)
    # opt = SGD(parameters, lr=0.01, momentum=0.7)

    backbone.to(device)

    bar = tqdm(range(epochs))

    xs, ys, _ = next(iter(dev_dataloader))
    xs, ys = xs.to(device), ys.to(device)

    for epoch in bar:
        corrects, tot = 0, 0
        c_corrects, c_tot = 0, 0

        backbone.train()

        epoch_losses = []

        for bi, (x, y, _) in enumerate(train_dataloader):

            x, y, _ = x.to(device), y.to(device), y.to(device)

            backbone_embs = backbone(x)
            batch_embs = model.forward_single_task(x, ti, False)

            centroids = calculate_centroids((xs, ys),
                                            model,
                                            ti)
            past_sim = 0
            dist = 0

            if ti > 0:
                for i in range(0, ti):

                    # past_centroids = calculate_centroids((x, y),
                    #                                 past_backbone,
                    #                                 past_embeddings[i],
                    #                                 classes,
                    #                                 gaussian=gaussian_centroids)
                    #
                    # current_centroids = calculate_centroids((x, y),
                    #                                 backbone,
                    #                                 embeddings_s[i],
                    #                                 classes,
                    #                                 gaussian=gaussian_centroids)

                    p_e = past_model.forward_single_task(x, i)
                    c_e = model.forward_single_task(x, i)

                    p_e_n = normalize(p_e)
                    c_e_n = normalize(c_e)

                    # p_e_n = normalize(past_centroids)
                    # c_e_n = normalize(current_centroids)

                    _dist = torch.norm(p_e_n - c_e_n, p=2, dim=1)
                    dist += _dist.mean()

                # dist = dist / ti
                dist = dist * 1

            # past_sim = 0
            # if ti > 0:
            #     # for i in range(0, ti):
            #     #     c_e = embeddings_s[i](backbone(x))
            #     #     c_e_n = normalize(c_e)
            #     #     te = normalize(tasks_centroids[i])
            #     #     sim = calculate_similarity(c_e_n, te)
            #     #     sim = 1 / (1 - sim)
            #     #     past_sim += sim.mean()
            #
            #     # if ti > 0:            #     # batch_embs
            #     #     # for i in range(0, ti):
            #     #     #     c_e = embeddings_s[i](backbone(x))
            #     #     #     _sim = -calculate_similarity(batch_embs, c_e)
            #     #     #     _sim = _sim / (2 * (1 ** 2))
            #     #     #     _sim = (-_sim).exp()
            #     #     #     # past_sim = 1 / (1 - past_sim)
            #     #     #     past_sim += _sim.sum(-1).mean()
            #     #
            #     #     past_centroids = torch.stack(classes_centroids, 0)
            #     #     # past_centroids = torch.cat((*tasks_centroids, centroids), 0)
            #
            #     past_centroids = normalize(torch.stack(classes_centroids, 0), -1)
            #     distance = -calculate_similarity(normalize(centroids, -1), past_centroids)
            #     # distance = distance / (2 * (1 ** 2))
            #     # distance = (-distance).exp()
            #     past_sim = 1 / (1 + past_sim)
            #     past_sim = past_sim.mean()

            sim = calculate_similarity(batch_embs, centroids)

            log_p_y = log_softmax(sim, dim=1)
            # loss_val = -log_p_y.gather(1, (y - min(classes)).unsqueeze(-1))
            loss_val = -log_p_y.gather(1, y.unsqueeze(-1))
            loss_val = loss_val.view(-1).mean()

            loss = loss_val + dist + past_sim

            opt.zero_grad(True)
            loss.backward()
            opt.step()

        train_centroids = calculate_centroids(dev_dataloader,
                                              model,
                                              ti)

        test_tot = 0
        test_corrects = 0

        for bi, (x, y, _) in enumerate(test_dataloader):
            x, y = x.to(device), y.to(device)
            # batch_embs = embeddings_s[ti](backbone(x))
            batch_embs = model.forward_single_task(x, ti, False)

            sim = calculate_similarity(batch_embs, train_centroids)

            sm = softmax(sim, -1)
            pred = torch.argmax(sm, 1)
            # test_corrects += (pred == (y - min(classes))).float().sum().item()
            test_corrects += (pred == y).float().sum().item()
            test_tot += pred.shape[0]

        task_score = test_corrects / test_tot
        bar.set_postfix({'test score': task_score})

    with torch.no_grad():
        train_centroids = calculate_centroids(dev_dataloader,
                                              model,
                                              ti)
    #     embeddings = []
    #
    #     for x, y, _ in dev_dataloader:
    #         batch_embs = embeddings_s[ti](backbone(x.to(device)))
    #
    #         dist = - calculate_similarity(batch_embs, train_centroids)
    #         embeddings.extend(batch_embs.cpu().detach().tolist())
    #
    #     embeddings = np.asarray(embeddings)
    #     print(embeddings.shape)
    #
    #     f = IsolationForest()
    #     f = LocalOutlierFactor(novelty=True, n_neighbors=50)
    #     f.fit(embeddings)
    #     forests.append(f)
    #
        classes_centroids.extend(train_centroids)
        tasks_centroids.append(train_centroids)
    #
        model.eval()

        labels = []

        for e_ti in range(0, ti + 1):

            test = test_stream[e_ti]

            classes = test.classes_in_this_experience

            test_dataloader = DataLoader(test.dataset,
                                         batch_size=64)

            centroids = tasks_centroids[e_ti]

            test_tot = 0
            test_corrects = 0

            offset = len(set(labels))

            for bi, (x, y, _) in enumerate(
                    tqdm(test_dataloader, leave=False,
                         total=len(test_dataloader))):

                labels.extend(y.cpu().tolist())

                x, y = x.to(device), y.to(device)

                batch_embs = model.forward_single_task(x, e_ti, False)

                sim = calculate_similarity(batch_embs, centroids)

                sm = softmax(sim, -1)
                pred = torch.argmax(sm, 1)
                test_corrects += (pred == (y - min(classes))) \
                    .float().sum().item()
                test_tot += pred.shape[0]

            task_score = test_corrects / test_tot
            tasks_matrix_scores[ti, e_ti] = task_score

        print(tasks_matrix_scores)
