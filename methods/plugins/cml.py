import random
from collections import defaultdict
from copy import deepcopy
from itertools import chain

import numpy as np
import torch
from avalanche.benchmarks.utils import AvalancheDataset, AvalancheSubset, \
    AvalancheConcatDataset
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.models import MultiTaskModule
from avalanche.training import BaseStrategy, ExperienceBalancedBuffer, \
    ClassBalancedBuffer
from avalanche.training.plugins import StrategyPlugin
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.special import erf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import SGDOneClassSVM
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.svm import OneClassSVM, SVC
from torch import cosine_similarity, log_softmax, softmax, nn, Tensor
from torch.nn import BatchNorm2d
from torch.nn.functional import normalize, dropout
from torch.nn.modules.dropout import _DropoutNd
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from torchvision.transforms import transforms


class Projector(nn.Module):
    def __init__(self, proj_type='offset', device=None):
        super().__init__()

        self.proj_type = proj_type
        self.device = device
        self.values = None

    def reset(self, n_tasks, embedding_size):
        if self.proj_type == 'offset':
            offsets = nn.ParameterList(
                [nn.Parameter(torch.randn(embedding_size))
                 for _ in range(n_tasks)])

            self.values = offsets

        elif self.proj_type == 'embeddings':
            emb = nn.Embedding(num_embeddings=n_tasks,
                               embedding_dim=embedding_size)
            linear = nn.Sequential(nn.Linear(embedding_size, embedding_size),
                                   nn.ReLU(),
                                   nn.Linear(embedding_size, embedding_size))
            params = nn.ModuleList([emb, linear])

            self.values = params
        elif self.proj_type == 'mlp':
            params = nn.ModuleList([nn.Sequential(
                nn.Linear(embedding_size, embedding_size),
                nn.ReLU(),
                nn.Linear(embedding_size, embedding_size))
                for _ in range(n_tasks)])

            self.values = params

    def forward(self, x, i):
        if self.proj_type == 'offset':
            return x + self.values[i]
        elif self.proj_type == 'embeddings':
            off = self.values[0](torch.tensor(i, dtype=torch.long))
            off = self.values[1](off)
            return x + off
        elif self.proj_type == 'mlp':
            return self.values[i](x)


class UncDropout(_DropoutNd):
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__(p, inplace)
        self._force_eval = False
        # self.training = True

    def force_eval(self):
        self._force_eval = True

    def _forward(self, input: Tensor) -> Tensor:
        return dropout(input, self.p, True, self.inplace)

    def _centroids_forward(self, input: Tensor) -> Tensor:
        return dropout(input, self.p, False, self.inplace)

    def forward(self, x):
        if self._force_eval:
            x = self._centroids_forward(x)
            self._force_eval = False
        else:
            x = self._forward(x)

        return x


def wrap_model(model: nn.Module, class_incremental=False):
    for name, module in model.named_children():
        if isinstance(module, BatchNorm2d):
            if class_incremental:
                setattr(model, name, ClassIncrementalBatchNorm2D(module))
            else:
                setattr(model, name, TaskIncrementalBatchNorm2D(module))
        else:
            wrap_model(module, class_incremental)


class TaskIncrementalBatchNorm2D(MultiTaskModule):
    def __init__(self,
                 bn: BatchNorm2d,
                 device=None):

        super().__init__()

        self.num_features = bn.num_features
        self.eps = bn.eps
        self.momentum = bn.momentum
        self.affine = bn.affine
        self.track_running_stats = bn.track_running_stats
        self.device = device

        self.bns = nn.ModuleDict()
        self.freeze_bn = {}

        self.current_task = None

    def train_adaptation(self, dataset: AvalancheDataset = None):

        cbn = len(self.bns)
        nbn = BatchNorm2d(self.num_features,
                          momentum=self.momentum,
                          affine=self.affine,
                          # device=self.device,
                          track_running_stats=self.track_running_stats,
                          eps=self.eps)

        self.bns[str(cbn)] = nbn

    def set_task(self, t):
        self.current_task = t

    def freeze_eval(self, t):
        self.freeze_bn[str(t)] = True
        self.bns[str(t)].eval()

        for p in self.bns[str(t)].parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode

        for k, v in self.bns.items():
            if not self.freeze_bn.get(k, False):
                v.train(mode)

        return self

    def eval(self):
        self.training = False
        for module in self.children():
            module.train(False)
        return self

    def forward_single_task(self, x: torch.Tensor,
                            task_label: int) -> torch.Tensor:

        return self.bns[str(task_label)](x)

    def forward(self, x, **kwargs):

        task_labels = self.current_task

        if isinstance(task_labels, int):
            # fast path. mini-batch is single task.
            return self.forward_single_task(x, task_labels)
        else:
            unique_tasks = torch.unique(task_labels)
            if len(unique_tasks) == 1:
                unique_tasks = unique_tasks.item()
                return self.forward_single_task(x, unique_tasks)

        assert False


class ClassIncrementalBatchNorm2D(MultiTaskModule):
    def __init__(self,
                 bn: BatchNorm2d,
                 device=None):

        super().__init__()

        # self.num_features = bn.num_features
        # self.eps = bn.eps
        # self.momentum = bn.momentum
        # self.affine = bn.affine
        # self.track_running_stats = bn.track_running_stats
        # self.device = device

        self.freeze = False

        self.bn = bn

    # def train_adaptation(self, dataset: AvalancheDataset = None):
    #
    #     cbn = len(self.bns)
    #     nbn = BatchNorm2d(self.num_features,
    #                       momentum=self.momentum,
    #                       affine=self.affine,
    #                       # device=self.device,
    #                       track_running_stats=self.track_running_stats,
    #                       eps=self.eps)
    #
    #     self.bns[str(cbn)] = nbn

    # def set_task(self, t):
    #     self.current_task = t

    def freeze_eval(self, t):
        self.freeze = True
        self.bn.eval()

        for p in self.bn.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode

        if not self.freeze:
            self.bn.train(mode)

        return self

    def eval(self):
        self.training = False
        for module in self.children():
            module.train(False)
        return self

    def forward_single_task(self, x: torch.Tensor,
                            task_label: int = None) -> torch.Tensor:

        return self.bn(x)

    def forward(self, x, **kwargs):
        return self.forward_single_task(x, None)


class BatchNormModelWrap(MultiTaskModule):
    def forward_single_task(self, x: torch.Tensor,
                            task_label: int, **kwargs) -> torch.Tensor:

        return self.model.forward_single_task(x, task_label)

    def __init__(self, model: nn.Module):
        super().__init__()

        self.model = model
        wrap_model(self.model)

    @property
    def feature_extractor(self):
        return self.model.feature_extractor

    def set_task(self, t):
        for module in self.model.modules():
            if isinstance(module, TaskIncrementalBatchNorm2D):
                module.set_task(t)

    def forward(self, x, task_labels, **kwargs):
        if not isinstance(task_labels, int):
            task_labels = torch.unique(task_labels)
            if len(task_labels) == 1:
                task_labels = task_labels.item()
            else:
                assert False

        self.set_task(task_labels)

        return self.model(x, task_labels, **kwargs)


class ClassIncrementalBatchNormModelWrap(MultiTaskModule):
    def forward_single_task(self, x: torch.Tensor,
                            task_label: int, **kwargs) -> torch.Tensor:

        return self.model.forward_single_task(x, task_label)

    def __init__(self, model: nn.Module):
        super().__init__()

        self.model = model
        wrap_model(self.model, True)

    @property
    def feature_extractor(self):
        return self.model.feature_extractor

    def forward(self, x, task_labels, **kwargs):
        if not isinstance(task_labels, int):
            task_labels = torch.unique(task_labels)
            if len(task_labels) == 1:
                task_labels = task_labels.item()
            else:
                assert False

        return self.model(x, task_labels, **kwargs)


# class ExperienceBalancedBuffer(BalancedExemplarsBuffer):
#     """ Rehearsal buffer with samples balanced over experiences.
#
#     The number of experiences can be fixed up front or adaptive, based on
#     the 'adaptive_size' attribute. When adaptive, the memory is equally
#     divided over all the unique observed experiences so far.
#     """
#
#     def __init__(self, max_size: int, adaptive_size: bool = True,
#                  num_experiences=None):
#         """
#         :param max_size: max number of total input samples in the replay
#             memory.
#         :param adaptive_size: True if mem_size is divided equally over all
#                               observed experiences (keys in replay_mem).
#         :param num_experiences: If adaptive size is False, the fixed number
#                                 of experiences to divide capacity over.
#         """
#         super().__init__(max_size, adaptive_size, num_experiences)
#
#     def update(self, strategy: "BaseStrategy", **kwargs):
#         new_data = strategy.experience.dataset
#         num_exps = strategy.clock.train_exp_counter + 1
#         lens = self.get_group_lengths(num_exps)
#
#         new_buffer = ReservoirSamplingBuffer(lens[-1])
#         new_buffer.update_from_dataset(new_data)
#         self.buffer_groups[num_exps - 1] = new_buffer
#
#         for ll, b in zip(lens, self.buffer_groups.values()):
#             b.resize(strategy, ll)

class ContinualMetricLearningPlugin(StrategyPlugin):
    def __init__(self, penalty_weight: float, sit=False,
                 sit_penalty_wights: float = 0.02,
                 sit_memory_size: int = 200,
                 num_experiences: int = 20):

        super().__init__()

        self.past_model = None
        self.penalty_weight = penalty_weight
        self.sit_penalty_wights = sit_penalty_wights

        self.similarity = 'euclidean'
        self.tasks_centroids = []
        self.thresholds = []

        self.memory = {}

        self.sit = sit
        self.tasks_forest = {}

        self.current_centroids = None

        self.storage_policy = ExperienceBalancedBuffer(
            max_size=sit_memory_size * num_experiences,
            adaptive_size=False, num_experiences=num_experiences)

    def custom_forward(self, model, x, task_id):
        # y = torch.cat([model(x, i) for i in range(task_id + 1)], -1)
        return model(x, task_id)

    def calculate_centroids(self, strategy: BaseStrategy, dataset):
        task = strategy.experience.current_experience

        # model = strategy.model
        device = strategy.device
        dataloader = DataLoader(dataset,
                                batch_size=strategy.train_mb_size)
        # batch_size=len(dataset))

        classes = set(dataset.targets)

        embs = defaultdict(list)

        # classes = set()
        # for x, y, tid in data:
        #     x, y, _ = d
        #     emb, _ = strategy.model.forward_single_task(x, t, True)
        #     classes.update(y.detach().cpu().tolist())

        classes = sorted(classes)

        for d in dataloader:
            x, y, tid = d
            x = x.to(device)
            # embeddings = strategy.model.forward_single_task(x, tid, False)

            # embeddings = strategy.model(x, tid)
            embeddings = self.custom_forward(strategy.model, x, task)

            for c in classes:
                embs[c].append(embeddings[y == c])

        embs = {c: torch.cat(e, 0) for c, e in embs.items()}

        centroids = torch.stack([torch.mean(embs[c], 0) for c in classes], 0)

        # if self.sit and len(self.tasks_centroids) > 0:
        #     centroids_means = torch.cat(self.tasks_centroids, 0).sum(0,
        #                                                              keepdims=True)
        #     centroids += centroids_means

        return centroids

    def calculate_similarity(self, x, y, similarity: str = None, sigma=1):
        if similarity is None:
            similarity = self.similarity

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
            sim = cosine_similarity(a, b, -1)
            dist = (sim + 1) / 2
        else:
            assert False

        return dist

    def _loss_f(self, x, y, centroids):
        sim = self.calculate_similarity(x, centroids)

        log_p_y = log_softmax(sim, dim=1)

        # if self.sit:
        #     loss_val = -log_p_y.gather(1, (y.unsqueeze(-1) - y.max()))
        # else:
        loss_val = -log_p_y.gather(1, y.unsqueeze(-1))

        return loss_val

    def loss(self, strategy, **kwargs):
        if not strategy.model.training:
            return -1

        centroids = self.calculate_centroids(strategy,
                                             strategy.experience.dev_dataset)

        if self.sit and len(self.tasks_centroids) > 0:
            c = torch.cat(self.tasks_centroids, 0)
            centroids = torch.cat((c, centroids), 0)

        self.current_centroids = centroids

        mb_output, y, x = strategy.mb_output, strategy.mb_y, strategy.mb_x

        mb_output = self.custom_forward(strategy.model, x,
                                        strategy.experience.current_experience)

        # sim = self.calculate_similarity(mb_output, centroids)
        #
        # log_p_y = log_softmax(sim, dim=1)
        # loss_val = -log_p_y.gather(1, y.unsqueeze(-1))
        # loss_val = loss_val.view(-1).mean()

        loss_val = self._loss_f(mb_output, y, centroids).view(-1).mean()

        return loss_val

    @torch.no_grad()
    def after_training_exp(self, strategy, **kwargs):
        tid = strategy.experience.current_experience

        centroids = self.calculate_centroids(strategy,
                                             strategy.experience.
                                             dev_dataset)

        self.tasks_centroids.append(centroids)

        if isinstance(strategy.model, BatchNormModelWrap):
            for name, module in strategy.model.named_modules():
                if isinstance(module, (TaskIncrementalBatchNorm2D,
                                       ClassIncrementalBatchNorm2D)):

                    module.freeze_eval(tid)
                    for p in module.parameters():
                        p.requires_grad_(False)

        # for param in strategy.model.classifier.classifiers[str(tid)].parameters():
        #     param.requires_grad_(False)

        if self.sit:
            self.storage_policy.update(strategy, **kwargs)

            # dataloader = DataLoader(strategy.experience.dataset,
            #                         batch_size=100)
            #
            # x, y, t = next(iter(dataloader))
            # self.memory[tid] = (x, y, t)

        #     embeddings = []
        #     for x, y, t in dataloader:
        #
        #         if tid not in self.memory:
        #             self.memory[tid] = (x, y, t)
        #
        #         x = x.to(strategy.device)
        #
        #         # e = strategy.model.forward_single_task(x, tid, False)
        #         # if self.sit:
        #         #     e, _ = strategy.model(strategy.mb_x, t, t=10)
        #         # else:
        #         e = self.past_model(x, t)
        #
        #         embeddings.extend(e.cpu().tolist())
        #
        #     embeddings = np.asarray(embeddings)
        #
        #     ln = len(embeddings)
        #     nn = 50 if ln > 100 else ln // 3
        #
        #     # f = LocalOutlierFactor(novelty=True,
        #     #                        n_neighbors=50,
        #     #                        contamination=0.1)
        #
        #     # f = OneClassSVM(nu=0.1, kernel='linear')
        #
        #     f = IsolationForest(n_estimators=500,
        #                         n_jobs=-1,
        #                         contamination=0.1,
        #                         max_samples=0.7)
        #
        #     f.fit(embeddings)
        #
        #     self.tasks_forest[tid] = f
        #
        #     dataloader = DataLoader(strategy.experience.dataset,
        #                             batch_size=strategy.train_mb_size)
        #
        #     # if self.sit and len(self.tasks_centroids) > 0:
        #     #     centroids = torch.cat(self.tasks_centroids, 0)
        #
        #     distances = []
        #     for x, y, _ in dataloader:
        #         x = x.to(strategy.device)
        #         y = y.to(strategy.device)
        #         # e = self.past_model(x, tid, t=10)[0]
        #         e = self.past_model(x, tid)
        #         sim = self.calculate_similarity(e, centroids)
        #         pred = sim.argmax(-1)
        #
        #         mask = pred == y
        #         sim = sim[mask]
        #         y = y[mask]
        #
        #         sim = sim.gather(1, y.unsqueeze(-1))
        #         distances.append(sim.cpu().numpy()[0])
        #
        #     distances = np.asarray(distances)
        #     a = np.quantile(distances,
        #                     list(np.arange(0.1, 1.0, 0.05)) + [0.99])[None, :]
        #
        #     if len(self.thresholds) == 0:
        #         self.thresholds = a
        #     else:
        #         self.thresholds = np.concatenate((self.thresholds, a), 0)
        else:
            self.past_model = deepcopy(strategy.model)
            self.past_model.eval()

    @torch.no_grad()
    def calculate_classes(self, strategy, embeddings):

        mode = strategy.model.training
        strategy.model.eval()

        correct_task = strategy.experience.current_experience
        y = strategy.mb_y

        n_classes_so_far = sum(t.shape[0] for t in self.tasks_centroids)
        # centroids = torch.cat(self.tasks_centroids, 0)

        if False and self.sit and len(self.tasks_centroids) > 1:
            distances = []
            predictions = []
            probs = []
            scores = []

            offsets = [0]

            for i in range(len(self.tasks_centroids)):
                if i > 0:
                    offsets.append(len(self.tasks_centroids[i]))

                # mask = np.ones(n_classes_so_far)
                # mask[offsets[i] + len(self.tasks_centroids[i]):] = 0
                # mask = mask[None, :]

                # c = torch.cat(self.tasks_centroids[i], 0)
                # e, em = strategy.model(strategy.mb_x, i, t=10)
                # e = strategy.model(strategy.mb_x, i)
                e = self.custom_forward(strategy.model, strategy.mb_x, i)

                # v = self.tasks_forest[i].decision_function(e.cpu().numpy())
                # scores.append(v)

                # all_sims.append(np.asarray([self.calculate_similarity(_e, torch.cat(self.tasks_centroids, 0)).cpu().numpy() for _e in em]))

                sim = self.calculate_similarity(e, self.tasks_centroids[i])

                sim = sim.cpu().numpy()
                distances.append(sim)
                # sim[:, offsets[i] + len(self.tasks_centroids[i]):] = -np.inf

                predictions.append(sim.argmax(1))

                # all_sims.append(sim)
                # sim = mask * sim + -np.inf * (1 - mask)

                sm = np.exp(sim) / np.exp(sim).sum(-1, keepdims=True)

                # sm = softmax(sim * mask, -1)

                probs.append(sm)

            offset = offsets[correct_task]
            distances = np.stack(distances, 1)
            predictions = np.asarray(predictions)
            probs = np.asarray(probs)
            scores = np.asarray(scores)

            probs = 1 - (distances / distances.sum(-1, keepdims=True))

            # counters = np.zeros_like(distances)
            #
            # for di, d in enumerate(distances):
            #     for i in range(len(d)):
            #         th = self.thresholds[i]
            #         for j in range(d.shape[-1]):
            #             counters[di, i, j] = (d[i, j] > th).sum()
            #
            # mx, mn = counters.max(-1), counters.min(-1)
            #
            # predicted_tasks = (mx - mn).argmax(-1)

            predicted_tasks = probs.max(-1).argmax(-1)
            correct_prediction_mask = predicted_tasks == correct_task

            pred = probs[:, correct_task].argmax(-1)
            # pred += offsets[correct_task]

            # pred = predictions[correct_task] + offsets[correct_task]
            pred = correct_prediction_mask * pred + (
                    1 - correct_prediction_mask) * -1

            # probs = np.exp(all_sims) / np.exp(all_sims).sum(-1, keepdims=True)
            # entropy = -(probs * np.log(probs)).sum(-1) / np.log(2)
            #
            # predicted_task = distances.max(-1).argmax(1)[None, :]
            #
            # pred = np.take_along_axis(predictions, predicted_task, 0)[0]

            pred = torch.tensor(pred,
                                device=strategy.device,
                                dtype=torch.long)

            # c = torch.cat(self.tasks_centroids, 0)
            #
            # sim = self.calculate_similarity(embeddings, c)
            # sm = softmax(sim, -1)
            # pred = torch.argmax(sm, 1)

            # correct_embs = None
            # ood_predictions = []
            #
            # ssims = []
            # _ssims = []
            # predictions = []
            #
            # for ti, forest in self.tasks_forest.items():
            #     e = strategy.model(strategy.mb_x, ti)
            #
            #     sim = self.calculate_similarity(e, self.tasks_centroids[ti])
            #     _ssims.append(sim.cpu().numpy())
            #
            #     sim = softmax(sim, -1).cpu().numpy()
            #
            #     ssims.append(sim.max(1))
            #     predictions.append(sim.argmax(1))
            #
            #     if ti == correct_task:
            #         correct_embs = e
            #
            #     e = e.cpu().numpy()
            #
            #     forest_prediction = forest.decision_function(e)
            #     # forest_prediction = forest.score_samples(e)
            #
            #     # mn, std = forest_prediction.mean(), forest_prediction.std()
            #     #
            #     # nv = forest_prediction - mn
            #     # nv = nv / (2**0.5 * std)
            #     # forest_prediction = erf(nv)
            #
            #     # forest_prediction = np.maximum(0, nv)
            #
            #     ood_predictions.append(forest_prediction)
            #
            # ood_predictions = np.asarray(ood_predictions)
            # # ood_predictions = ood_predictions + 0.5
            #
            # ssims = np.asarray(ssims)
            # _ssims = np.asarray(_ssims)
            # predictions = np.asarray(predictions)
            #
            # # mn, std = predictions.mean(1, keepdims=True), \
            # #           predictions.std(1, keepdims=True)
            # #
            # # nv = predictions - mn
            # # nv = nv / ((2 ** 0.5) * std)
            # # predictions = erf(nv)
            #
            # predicted_tasks = np.argmax(ood_predictions, 0)
            # # predicted_tasks = np.argmax(ssims, 0)
            #
            # correct_prediction_mask = predicted_tasks == correct_task
            #
            # pred = correct_prediction_mask * predictions[correct_task] + (
            #         1 - correct_prediction_mask) * -1
            #
            # pred = torch.tensor(pred,
            #                     device=strategy.device,
            #                     dtype=torch.long)

            # correct_prediction_mask = torch.tensor(correct_prediction_mask,
            #                                        device=strategy.device,
            #                                        dtype=torch.long)
            #
            # centroids = self.tasks_centroids[correct_task]
            # sim = self.calculate_similarity(correct_embs, centroids)
            # sm = softmax(sim, -1)
            # sm = torch.argmax(sm, 1)
            #
            # pred = correct_prediction_mask * sm + (
            #         1 - correct_prediction_mask) * -1
        else:
            if self.sit:
                # embeddings, _ = strategy.model(strategy.mb_x, correct_task, t=10)
                centroids = torch.cat(self.tasks_centroids, 0)
            else:
                centroids = self.tasks_centroids[correct_task]

            sim = self.calculate_similarity(embeddings, centroids)
            sm = softmax(sim, -1)
            pred = torch.argmax(sm, 1)

        strategy.model.train(mode)

        return pred

    # def before_forward(self, strategy: 'BaseStrategy', **kwargs):
    #
    #     if len(self.memory) > 0 and self.sit:
    #         m_x, m_y, m_t = [], [], []
    #         # labels = set(strategy.experience.dataset.targets)
    #
    #         for tt, (x, y, t) in self.memory.items():
    #             m_t.extend(t)
    #             m_y.extend(y)
    #             m_x.extend(x)
    #
    #             # for k, v in m.items():
    #             #     m_t.extend([t] * len(v))
    #             #     m_y.extend([k] * len(v))
    #             #     m_x.extend(v)
    #
    #         x, y, t = strategy.mbatch
    #
    #         if len(m_x) > 0:
    #             ln = len(strategy.mb_x)
    #
    #             if len(m_x) > ln:
    #                 indexes = random.choices(range(len(m_x)), k=ln)
    #                 m_x = [m_x[i] for i in indexes]
    #                 m_y = [m_y[i] for i in indexes]
    #                 m_t = [m_t[i] for i in indexes]
    #
    #                 m_x = torch.stack(m_x, 0).to(strategy.device)
    #                 m_y = torch.tensor(m_y, dtype=torch.long,
    #                                    device=strategy.device)
    #                 m_t = torch.tensor(m_t, dtype=torch.long,
    #                                    device=strategy.device)
    #
    #             #     #     ln = len(m_x)
    #             # #
    #             # # if ln != len(m_x):
    #             # #     m_x = torch.stack([m_x[i] for i in indexes], 0).to(
    #             # #         strategy.device)
    #             # #     m_y = torch.tensor([m_y[i] for i in indexes],
    #             # #                        dtype=torch.long,
    #             # #                        device=strategy.device)
    #             # #     m_t = torch.tensor([m_t[i] for i in indexes],
    #             # #                        dtype=torch.long,
    #             # #                        device=strategy.device)
    #             # # else:
    #             #
    #                 mask = torch.bernoulli(torch.full((ln, ), 0.5,
    #                                                   device=strategy.device))
    #
    #                 x = m_x * mask[:, None, None, None] \
    #                     + (1 - mask[:, None, None, None]) * x
    #                 y = m_y * mask + (1 - mask) * y
    #                 y = y.type(torch.long)
    #                 t = m_t * mask + (1 - mask) * t
    #             else:
    #
    #                 m_x = torch.stack(m_x, 0).to(strategy.device)
    #                 m_y = torch.tensor(m_y, dtype=torch.long,
    #                                    device=strategy.device)
    #                 m_t = torch.tensor(m_t, dtype=torch.long,
    #                                    device=strategy.device)
    #
    #                 x = torch.cat((x, m_x), 0)
    #                 y = torch.cat((y, m_y), 0)
    #                 t = torch.cat((t, m_t), 0)
    #
    #             strategy.mbatch = [x, y, t]

    def before_backward(self, strategy, **kwargs):
        if self.sit:
            return

        if strategy.clock.train_exp_counter > 0:

            # strategy.model.eval()

            x, y, tdi = strategy.mbatch
            dist = 0
            tot_sim = 0

            # # if self.sit:
            dataloader = DataLoader(strategy.experience.dev_dataset,
                                    batch_size=len(x),
                                    # batch_size=len(
                                    #     strategy.experience.dev_dataset),
                                    shuffle=True)

            dev_x, dev_y, dev_t = next(iter(dataloader))

            dev_x, dev_y = dev_x.to(strategy.device), \
                           dev_y.to(strategy.device)

            if self.sit:
                for i in range(len(self.tasks_centroids)):
                    xp, yp, tp = self.memory[i]

                    loss = self._loss_f(
                        self.custom_forward(strategy.model, xp, i),
                        yp,
                        self.tasks_centroids[i])

                    dist = loss.view(-1).mean(0)

                    # xp = xp.to(strategy.device)
                    #
                    # p_e = self.custom_forward(self.past_model, xp, i)
                    # c_e = self.custom_forward(strategy.model, xp, i)
                    #
                    # # p_e = self.past_model(xp, i)
                    # # c_e = strategy.model(xp, i)
                    #
                    # _dist = torch.norm(p_e - c_e, p=2, dim=1)
                    # dist += _dist.mean()

                    # dev_e = strategy.model(x, i)
                    # dev_e = self.custom_forward(strategy.model, x, i)
                    #
                    # d = -self.calculate_similarity(dev_e, self.tasks_centroids[i])
                    #
                    # sim = 1 / (1 + d)
                    #
                    # tot_sim += sim.mean(1).mean(0)

                    # loss_val = self._loss_f(dev_e, y,
                    #                         self.current_centroids) \
                    #     .view(-1).mean()
                    #
                    # tot_sim += loss_val

                # dist = dist * self.penalty_weight
                # tot_sim = tot_sim * self.sit_penalty_wights
                #
                # if strategy.clock.train_exp_counter > 0:
                #     print(tot_sim, dist, strategy.loss)
                #
                # strategy.loss += dist + tot_sim

            else:
                for i in range(len(self.tasks_centroids)):
                    p_e = self.past_model(x, i)
                    c_e = strategy.model(x, i)

                    _dist = torch.norm(p_e - c_e, p=2, dim=1)
                    dist += _dist.mean()

            # if self.sit:
            #
            #     centroids = self.current_centroids
            #     past_centroids = torch.cat(self.tasks_centroids, 0)
            #     d = -self.calculate_similarity(centroids, past_centroids)
            #     sim = 1 / (1 + d)
            #
            #     tot_sim += sim.mean(1).sum()

            #     e = strategy.model(x, i)
            #
            #     d = -self.calculate_similarity(e, self.tasks_centroids[i])
            #
            #     sim = 1 / (1 + d)
            #
            #     tot_sim += sim.mean(1).mean(0)

            # tot_sim = tot_sim / len(self.tasks_centroids)
            # dist = dist / len(self.tasks_centroids)

            # if strategy.clock.train_exp_counter > 0:
            #     print(tot_sim, dist, strategy.loss)
            # strategy.model.train()

            dist = dist * self.penalty_weight
            tot_sim = tot_sim * self.sit_penalty_wights

            strategy.loss += dist + tot_sim

    def before_training_exp(self, strategy: "BaseStrategy",
                            num_workers: int = 0, shuffle: bool = True,
                            **kwargs):
        """
        Dataloader to build batches containing examples from both memories and
        the training dataset
        """

        if len(self.storage_policy.buffer) == 0:
            # first experience. We don't use the buffer, no need to change
            # the dataloader.
            return

        strategy.dataloader = ReplayDataLoader(
            strategy.adapted_dataset,
            self.storage_policy.buffer,
            oversample_small_tasks=True,
            num_workers=num_workers,
            batch_size=strategy.train_mb_size,
            shuffle=shuffle)


class DropContinualMetricLearningPlugin(StrategyPlugin):
    def __init__(self, penalty_weight: float, sit=False,
                 sit_penalty_wights: float = 0,
                 sit_memory_size: int = 200,
                 num_experiences: int = 20):

        super().__init__()

        self.scaler = Projector('mlp')
        self.forests = {}
        self.classifier = None
        self.past_model = None
        self.penalty_weight = penalty_weight
        self.sit_penalty_wights = sit_penalty_wights
        self.sit_memory_size = sit_memory_size

        self.similarity = 'euclidean'
        self.tasks_centroids = []
        self.thresholds = []

        self.memory = {}

        self.sit = sit
        self.tasks_forest = {}

        self.current_centroids = None
        self.initial_model = None

        self.storage_policy = ExperienceBalancedBuffer(
            max_size=sit_memory_size,
            adaptive_size=True)

    def custom_forward(self, model, x, task_id, t=None, force_eval=False):
        # y = torch.cat([model(x, i) for i in range(task_id + 1)], -1)

        # if force_eval and self.sit:
        #     model.classifier.classifiers[str(task_id)].force_eval()
        #     t = None
        #
        # return model(x, task_id, t=t)

        f = model(x, task_id)
        if t is None:
            return f

        return f, [f]

    def calculate_centroids(self, strategy: BaseStrategy, dataset, model=None,
                            task=None):
        if model is None:
            model = strategy.model

        if task is None:
            task = strategy.experience.current_experience

        # if self.sit:
        #     strategy.model.classifier.classifiers[str(task)].force_eval()

        # model = strategy.model
        device = strategy.device
        dataloader = DataLoader(dataset,
                                batch_size=strategy.train_mb_size)
        # batch_size=len(dataset))

        classes = set(dataset.targets)

        embs = defaultdict(list)

        # classes = set()
        # for x, y, tid in data:
        #     x, y, _ = d
        #     emb, _ = strategy.model.forward_single_task(x, t, True)
        #     classes.update(y.detach().cpu().tolist())

        classes = sorted(classes)

        for d in dataloader:
            x, y, tid = d
            x = x.to(device)

            embeddings = self.custom_forward(model, x, task,
                                             force_eval=True)

            for c in classes:
                embs[c].append(embeddings[y == c])

        embs = {c: torch.cat(e, 0) for c, e in embs.items()}

        centroids = torch.stack([torch.mean(embs[c], 0) for c in classes], 0)

        # if self.sit and len(self.tasks_centroids) > 0:
        #     centroids_means = torch.cat(self.tasks_centroids, 0).sum(0,
        #                                                              keepdims=True)
        #     centroids += centroids_means

        return centroids

    def calculate_similarity(self, x, y, distance: str = None, sigma=1):
        if distance is None:
            distance = self.similarity

        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        if d != y.size(1):
            raise Exception

        a = x.unsqueeze(1).expand(n, m, d)
        b = y.unsqueeze(0).expand(n, m, d)

        if distance == 'euclidean':
            similarity = -torch.pow(a - b, 2).sum(2).sqrt()
        elif distance == 'rbf':
            similarity = -torch.pow(a - b, 2).sum(2).sqrt()
            similarity = similarity / (2 * sigma ** 2)
            similarity = torch.exp(similarity)
        elif distance == 'cosine':
            similarity = cosine_similarity(a, b, -1)
        else:
            assert False

        return similarity

    def _loss_f(self, x, y, centroids):
        sim = self.calculate_similarity(x, centroids)

        log_p_y = log_softmax(sim, dim=1)

        # if self.sit:
        #     loss_val = -log_p_y.gather(1, (y.unsqueeze(-1) - y.max()))
        # else:
        loss_val = -log_p_y.gather(1, y.unsqueeze(-1))

        return loss_val

    def loss(self, strategy, **kwargs):
        if not strategy.model.training:
            return -1

        centroids = self.calculate_centroids(strategy,
                                             strategy.experience.dev_dataset)

        # if self.sit and len(self.tasks_centroids) > 0:
        #     c = torch.cat(self.tasks_centroids, 0)
        #     centroids = torch.cat((c, centroids), 0)

        self.current_centroids = centroids

        mb_output, y, x = strategy.mb_output, strategy.mb_y, strategy.mb_x

        mb_output = self.custom_forward(strategy.model, x,
                                        strategy.experience.current_experience)

        loss_val = self._loss_f(mb_output, y, centroids).view(-1).mean()

        return loss_val

    # def before_training(self, strategy: 'BaseStrategy', **kwargs):
    #     if self.initial_model is None:
    #         self.initial_model = deepcopy(strategy.model.feature_extractor.state_dict())
    #
    # def before_train_dataset_adaptation(self, strategy: 'BaseStrategy',
    #                                     **kwargs):
    #     strategy.model.feature_extractor.load_state_dict(self.initial_model)

    def after_training_exp(self, strategy, **kwargs):

        with torch.no_grad():
            tid = strategy.experience.current_experience

            centroids = self.calculate_centroids(strategy,
                                                 strategy.experience.
                                                 dev_dataset)

            self.tasks_centroids.append(centroids)

            if isinstance(strategy.model, BatchNormModelWrap):
                for name, module in strategy.model.named_modules():
                    # if isinstance(module, (TaskIncrementalBatchNorm2D,
                    #                        ClassIncrementalBatchNorm2D)):
                    if isinstance(module, TaskIncrementalBatchNorm2D):
                        module.freeze_eval(tid)

            self.past_model = deepcopy(strategy.model)
            self.past_model.eval()

            # for m in strategy.model.feature_extractor.modules():
            #     if hasattr(m, 'reset_parameters'):
            #         m.reset_parameters()

            self.storage_policy.update(strategy, **kwargs)

            if self.sit:
                # losses = []
                #
                # for x, y, t in DataLoader(strategy.experience.dataset, 64):
                #     x, y = x.to(strategy.device), y.to(strategy.device)
                #
                #     o = self.custom_forward(self.past_model, x, tid)
                #     loss = self._loss_f(o, y, self.tasks_centroids[tid])
                #     losses.extend(loss.detach().cpu().numpy())
                #
                # losses = np.concatenate(losses)
                # s = np.argsort(losses)

                idxs = np.arange(len(strategy.experience.dataset))
                # idxs = idxs[np.concatenate((s[:self.sit_memory_size // 2],
                #                             s[-self.sit_memory_size // 2:]))]
                np.random.shuffle(idxs)
                idxs = idxs[:self.sit_memory_size]

                dataset = AvalancheSubset(strategy.experience.dataset, idxs)

                self.memory[len(self.memory)] = dataset

                if len(self.memory) > 1:
                    X, Y = [], []

                    if isinstance(self.past_model, BatchNormModelWrap):
                        self.past_model.set_task(len(self.memory) - 1)

                    for k, v in self.memory.items():

                        v = v.eval()
                        Y.extend([k] * len(v))

                        for _x, _, _ in v:
                            _x = _x.to(strategy.device)
                            emb = self.past_model.feature_extractor(
                                _x[None, :]).view(-1).detach().cpu().numpy()
                            X.append(emb)
                            # X.append(_x)

                    X = np.asarray(X)
                    Y = np.asarray(Y)

                    svc = RandomForestClassifier(n_jobs=-1)
                    parameters = {'n_estimators': [10, 50, 100, 250, 1000],
                                  'max_depth': [None, 3, 5, 15, 20]}

                    svc = SVC()
                    parameters = {'kernel': ['linear', 'rbf'],
                                  'C': [0.01, 1, 10, 100, 1000],
                                  'gamma': ['scale', 'auto']}

                    # svc = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
                    # parameters = {'n_neighbors': [1, 3, 5, 10, 15, 20, 50],
                    #               'p': [1, 2]}

                    # clf = GridSearchCV(svc, parameters, n_jobs=-1,
                    #                    verbose=True, cv=5)
                    # clf.fit(X, Y)
                    # # print(clf.cv_results_)
                    # svc = clf.best_estimator_
                    # svc.fit(X, Y)

                    self.classifier = svc

                    # score = svc.score(X, Y)
                    # print(score)

                    pca_X = PCA(n_components=2).fit_transform(X)
                    plt.scatter(pca_X[:, 0], pca_X[:, 1], c=Y)
                    plt.show()

                # for k, v in self.memory.items():
                #     embeddings = []
                #
                #     dataloader = DataLoader(v,
                #                             batch_size=strategy.train_mb_size)
                #
                #     for x, y, t in dataloader:
                #
                #         # if tid not in self.memory:
                #         #     self.memory[tid] = (x, y, t)
                #
                #         x = x.to(strategy.device)
                #
                #         # e = strategy.model.forward_single_task(x, tid, False)
                #         # if self.sit:
                #         #     e, _ = strategy.model(strategy.mb_x, t, t=10)
                #         # else:
                #         e = self.past_model(x, t)
                #
                #         embeddings.extend(e.cpu().tolist())
                #
                #     embeddings = np.asarray(embeddings)
                #
                #     ln = len(embeddings)
                #     nn = 50 if ln > 100 else ln // 3
                #
                #     f = LocalOutlierFactor(novelty=True,
                #                            n_neighbors=50,
                #                            contamination=0.05)
                #
                #     # f = OneClassSVM(nu=0.1, kernel='linear')
                #
                #     f = IsolationForest(n_estimators=500,
                #                         n_jobs=-1,
                #                         contamination=0.1)
                #
                #     f.fit(embeddings)
                #
                #     self.forests[k] = f

        # if self.sit:
        #     emb_shape = self.tasks_centroids[0].shape[1]
        #     # self.scaler.append(nn.Parameter(torch.randn(emb_shape)))
        #
        #     if len(self.memory) > 1:
        #
        #         xs, ys, ts = [], [], []
        #
        #         offsets = np.cumsum([len(c) for c in self.tasks_centroids])
        #         offsets_tensor = torch.tensor([0] + offsets.tolist(),
        #                                       dtype=torch.long,
        #                                       device=strategy.device)
        #
        #         # for k, v in self.memory.items():
        #         #     if k == 0:
        #         #         off = 0
        #         #     else:
        #         #         off = offsets[k - 1]
        #         #
        #         #     for x, y, t in v.eval():
        #         #         xs.append(x)
        #         #         ys.append(y + off)
        #         #         ts.append(t)
        #         #
        #         # xs = torch.stack(xs, 0)
        #         # ys = torch.tensor(ys, dtype=torch.long)
        #         # ts = torch.tensor(ts, dtype=torch.long)
        #         #
        #         # concatenated_tasks = TensorDataset(xs, ys, ts)
        #
        #         # concatenated_tasks = AvalancheConcatDataset(self.storage_policy.buffer)
        #         concatenated_tasks = self.storage_policy.buffer
        #         # concatenated_tasks = ConcatDataset(self.memory.values())
        #
        #         n = len(concatenated_tasks)
        #         n_test = int(0.1 * n)
        #         idxs = np.arange(n)
        #         np.random.shuffle(idxs)
        #
        #         test_dataset = AvalancheSubset(concatenated_tasks,
        #                                        idxs[:n_test])
        #         concatenated_tasks = AvalancheSubset(concatenated_tasks,
        #                                              idxs[n_test:])
        #
        #         loader = DataLoader(concatenated_tasks,
        #                             batch_size=10,
        #                             shuffle=True)
        #
        #         test_loader = DataLoader(test_dataset.eval(),
        #                                  batch_size=10,
        #                                  shuffle=True)
        #
        #         # scaler = nn.Embedding(num_embeddings=len(self.tasks_centroids),
        #         #                       embedding_dim=
        #         #                       self.tasks_centroids[0].shape[1])
        #
        #         num_tasks = len(self.tasks_centroids)
        #
        #         # scaler = nn.ParameterList()
        #         # scaler.extend([nn.Parameter(torch.randn(emb_shape))
        #         # for _ in range(num_tasks)])
        #
        #         self.scaler.reset(num_tasks, emb_shape)
        #
        #         self.scaler = self.scaler.to(strategy.device)
        #
        #         opt = Adam(chain(self.scaler.parameters()), 1e-3)
        #
        #         strategy.model.eval()
        #
        #         best_model = None
        #         best_score = 0
        #
        #         for task in range(50):
        #             corrects = 0
        #             total = 0
        #
        #             for x, y, t in loader:
        #                 x, y, t = x.to(strategy.device), \
        #                           y.to(strategy.device), \
        #                           t.to(strategy.device)
        #
        #                 e = 0
        #
        #                 y += torch.index_select(offsets_tensor, 0, t)
        #
        #                 for task in range(num_tasks):
        #                     # e += self.custom_forward(strategy.model, x, task) \
        #                     #      + self.scaler[task]
        #
        #                     e += self.scaler(
        #                         self.custom_forward(strategy.model, x, task),
        #                         task)
        #
        #                 # centroids = torch.cat([c + self.scaler[task]
        #                 #                        for task, c in
        #                 #                        enumerate(self.tasks_centroids)],
        #                 #                       0)
        #
        #                 centroids = torch.cat([self.scaler(c, task)
        #                                        for task, c in
        #                                        enumerate(self.tasks_centroids)],
        #                                       0)
        #
        #                 loss = self._loss_f(e, y, centroids)
        #                 loss = loss.view(-1).mean()
        #
        #                 opt.zero_grad()
        #                 loss.backward()
        #                 opt.step()
        #
        #             for x, y, t in test_loader:
        #                 x, y, t = x.to(strategy.device), \
        #                           y.to(strategy.device), \
        #                           t.to(strategy.device)
        #
        #                 e = 0
        #
        #                 y += torch.index_select(offsets_tensor, 0, t)
        #
        #                 for task in range(num_tasks):
        #                     # e += self.custom_forward(strategy.model, x, task) \
        #                     #      + self.scaler[task]
        #
        #                     e += self.scaler(
        #                         self.custom_forward(strategy.model, x, task),
        #                         task)
        #
        #                 # centroids = torch.cat([c + self.scaler[task]
        #                 #                        for task, c in
        #                 #                        enumerate(self.tasks_centroids)],
        #                 #                       0)
        #
        #                 centroids = torch.cat([self.scaler(c, task)
        #                                        for task, c in
        #                                        enumerate(self.tasks_centroids)],
        #                                       0)
        #
        #                 pred = self.calculate_similarity(e, centroids).argmax(
        #                     -1)
        #                 corrects += (pred == y).sum().item()
        #                 total += len(pred)
        #
        #             test_score = corrects / total
        #             if test_score > best_score:
        #                 best_score = test_score
        #                 best_model = (deepcopy(self.scaler.state_dict()),
        #                               deepcopy(strategy.model.state_dict()))
        #
        #             print(corrects, total)
        #
        #         scaler_dict, model_dict = best_model
        #
        #         self.scaler.load_state_dict(scaler_dict)
        #         strategy.model.load_state_dict(model_dict)
        #
        #         # self.scaler = scaler
        #     # dataloader = DataLoader(strategy.experience.dataset,
        #     #                         batch_size=strategy.train_mb_size)
        #     #
        #     # distances = []
        #     # entropies = []
        #
        #     # for x, y, _ in dataloader:
        #     #     x = x.to(strategy.device)
        #     #     y = y.to(strategy.device)
        #     #     # e = self.past_model(x, tid, t=10)[0]
        #     #     e, _ = self.custom_forward(self.past_model, x, tid,
        #     #                                force_eval=False,
        #     #                                t=10)
        #     #
        #     #     # e = self.past_model(x, tid)
        #     #     sim = self.calculate_similarity(e, centroids)
        #     #     probs = torch.softmax(sim, -1)
        #     #     pred = probs.argmax(-1)
        #     #
        #     #     entropy = -(probs.log() * probs).sum(-1) / \
        #     #               np.log(probs.shape[-1])
        #     #
        #     #     mask = pred == y
        #     #     sim = sim[mask]
        #     #     y = y[mask]
        #     #     entropy = entropy[mask]
        #     #
        #     #     sim = sim.gather(1, y.unsqueeze(-1))
        #     #     distances.extend(sim.cpu().numpy()[0])
        #     #     entropies.extend(entropy.cpu().numpy())
        #     #
        #     # distances = np.asarray(distances)
        #     # entropies = np.asarray(entropies)
        #     #
        #     # q3 = np.quantile(entropies, 0.75)
        #     # q1 = np.quantile(entropies, 0.25)
        #     #
        #     # thres = q3 - 0.8 * (q3 - q1)
        #     # self.thresholds.append((q1, q3, thres))
        #
        #     # a = np.quantile(distances,
        #     #                 list(np.arange(0.1, 1.0, 0.05)) + [0.99])[None, :]
        #     #
        #     # if len(self.thresholds) == 0:
        #     #     self.thresholds = a
        #     # else:
        #     #     self.thresholds = np.concatenate((self.thresholds, a), 0)

    def before_train_dataset_adaptation(self, strategy: 'BaseStrategy',
                                        **kwargs):
        # for m in strategy.model.modules():
        #     if hasattr(m, 'reset_parameters'):
        #         m.reset_parameters()

        num_tasks = len(self.tasks_centroids)
        if num_tasks == 0:
            return

        emb_shape = self.tasks_centroids[0].shape[1]

        self.scaler.reset(num_tasks + 1, emb_shape)
        self.scaler = self.scaler.to(strategy.device)

        strategy.optimizer.state = defaultdict(dict)
        strategy.optimizer.param_groups[0]['params'] = list(
            chain(strategy.model.parameters(),
                  self.scaler.parameters()))

    def _transform_predict(self, strategy, current_task, n):
        T = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ])

        distances = []

        for _ in range(n):
            x = T(strategy.mb_x)
            me, ae = self.custom_forward(strategy.model,
                                         x, current_task,
                                         t=10)
            sim = self.calculate_similarity(me,
                                            self.tasks_centroids[current_task])

            distances.append(sim)

        return torch.stack(distances, 1)

    def _predict(self, strategy, current_task):
        me = self.custom_forward(strategy.model,
                                 strategy.mb_x, current_task)

        sim = self.calculate_similarity(me,
                                        self.tasks_centroids[current_task])

        return sim.argmax(-1)

    @torch.no_grad()
    def calculate_classes(self, strategy, embeddings):
        # def loss_f(logits):
        #     return -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(
        #         1).mean()

        # def get_function_to_minimize():
        #     embs = []
        #
        #     for i in range(len(self.tasks_centroids)):
        #         me, ae = self.custom_forward(strategy.model,
        #                                      strategy.mb_x, i, t=10)
        #         embs.append(me)
        #
        #     def closure(alphas):
        #         tot_loss = 0
        #         alphas = torch.tensor(alphas, device=strategy.device)
        #
        #         for i in range(len(self.tasks_centroids)):
        #             sim = self.calculate_similarity(embs[i] * alphas[i],
        #                                             self.tasks_centroids[i])
        #
        #             loss = loss_f(sim)
        #             tot_loss += loss
        #
        #         return tot_loss.item()
        #
        #     return closure
        #
        # def simulated_annealing(objective, n_iterations, step_size,
        #                         temp, x0):
        #     best = x0
        #     # best = np.exp(best) / np.exp(best).sum(0, keepdims=True)
        #
        #     # evaluate the initial point
        #     best_eval = objective(best)
        #     # current working solution
        #     curr, curr_eval = best, best_eval
        #     # run the algorithm
        #     for i in range(n_iterations):
        #         # take a step
        #         candidate = curr + np.random.rand(*best.shape) * step_size
        #         _candidate = np.exp(candidate) / np.exp(candidate).sum(0,
        #                                                                keepdims=True)
        #         # evaluate candidate point
        #         candidate_eval = objective(_candidate)
        #         # check for new best solution
        #         if candidate_eval < best_eval:
        #             # store new best point
        #             best, best_eval = candidate, candidate_eval
        #             # report progress
        #             # print('>%d = %.5f' % (i, best_eval))
        #         # difference between candidate and current point evaluation
        #         diff = candidate_eval - curr_eval
        #         # calculate temperature for current epoch
        #         t = temp / float(i + 1)
        #         # calculate metropolis acceptance criterion
        #         metropolis = np.exp(-diff / t)
        #         # check if we should keep the new point
        #         if diff < 0 or np.random.rand() < metropolis:
        #             # store the new current point
        #             curr, curr_eval = candidate, candidate_eval
        #
        #     return best, best_eval

        mode = strategy.model.training
        strategy.model.eval()

        correct_task = strategy.experience.current_experience
        y = strategy.mb_y
        x = strategy.mb_x

        n_classes_so_far = sum(t.shape[0] for t in self.tasks_centroids)
        # centroids = torch.cat(self.tasks_centroids, 0)

        if self.sit and len(self.tasks_centroids) > 1:
            cumsum = np.cumsum([len(c) for c in self.tasks_centroids])

            upper = cumsum[correct_task]
            lower = 0 if correct_task == 0 else cumsum[correct_task - 1]

            emb_shape = self.tasks_centroids[0].shape[1]
            num_tasks = len(self.tasks_centroids)

            e = 0

            for task in range(num_tasks):
                # e += self.custom_forward(strategy.model, x, task) \
                #      + self.scaler[task]

                e += self.scaler(self.custom_forward(strategy.model, x, task),
                                 task)

            # centroids = torch.cat([c + self.scaler[task]
            #                        for task, c in
            #                        enumerate(self.tasks_centroids)],
            #                       0)

            centroids = torch.cat([self.scaler(c, task)
                                   for task, c in
                                   enumerate(self.tasks_centroids)],
                                  0)

            pred = self.calculate_similarity(e, centroids).argmax(-1)
            pred[pred >= upper] = -1
            pred = pred - lower

            # if isinstance(self.past_model, BatchNormModelWrap):
            #     self.past_model.set_task(len(self.memory) - 1)

            # scores = []
            # for k, f in self.forests.items():
            #     scores.append(f.decision_function(
            #         strategy.model(strategy.mb_x, k).detach().cpu().numpy()))
            #
            # scores = np.asarray(scores)
            # predicted_tasks = scores.argmax(0)

            # emb = strategy.model.feature_extractor(strategy.mb_x)
            # emb = emb.view(strategy.mb_x.shape[0], -1).detach().cpu().numpy()
            #
            # predicted_tasks = self.classifier.predict(emb)

            # alphas = np.full((len(self.tasks_centroids), len(y), 1),
            #                  1 / len(self.tasks_centroids))
            #
            # alphas, best = simulated_annealing(get_function_to_minimize(),
            #                                    2000,
            #                                    0.2,
            #                                    50,
            #                                    alphas)
            #
            # best_alpha = int(alphas.argmax(0)[0, 0])
            #
            # # print(best_alpha)
            #
            # me, ae = self.custom_forward(strategy.model,
            #                              strategy.mb_x, best_alpha, t=10)
            #
            # sim = self.calculate_similarity(me,
            #                                 self.tasks_centroids[best_alpha])
            #
            # pred = sim.argmax(1)

            # alphas = np.full((len(self.tasks_centroids), len(y), 1),
            #                  1 / len(self.tasks_centroids))
            #
            # a = np.full(len(self.tasks_centroids),
            #                                     1 / len(self.tasks_centroids))
            #
            # for i in range(100):
            #     l = to_minimize(a)
            #     print(l, a)
            #
            #     a[np.random.choice(len(a))] *= 1.5
            #     # a = np.random.rand(len(a))
            #     a = np.exp(a) / np.exp(a).sum()
            #
            # a[1] = 10
            # res = minimize(to_minimize, a, method='Newton-CG')

            # offsets = [0]
            #
            # scores = []
            # distances = []
            # predictions = []
            #
            # for i in range(len(self.tasks_centroids)):
            #     if i > 0:
            #         offsets.append(len(self.tasks_centroids[i]))
            #
            #     # diss = self._transform_predict(strategy, i, 10)
            #     diss = self._predict(strategy, i)
            #
            #     probs = softmax(diss, -1)
            #     entropy = -(probs.log() * probs).sum(-1) / \
            #               np.log(probs.shape[-1])
            #
            #     scores.append(entropy.cpu().numpy())
            #     distances.append(diss.cpu().numpy())
            #
            #     # prediction = diss.mean(1).argmax(-1)
            #     prediction = diss.argmax(-1)
            #     predictions.append(prediction.cpu().numpy())
            #
            # #     if best_value is None or entropy < best_value:
            # #         best_value = entropy
            # #         selected_task = i
            # #         predicted_class = distances.mean(1).argmax(-1)
            # #
            # #
            # #     # mask = np.ones(n_classes_so_far)
            # #     # mask[offsets[i] + len(self.tasks_centroids[i]):] = 0
            # #     # mask = mask[None, :]
            # #
            # #     # c = torch.cat(self.tasks_centroids[i], 0)
            # #     # e, em = strategy.model(strategy.mb_x, i, t=10)
            # #     # e = strategy.model(strategy.mb_x, i)
            # #     me, ae = self.custom_forward(strategy.model,
            # #                                  strategy.mb_x, i, t=10)
            # #
            # #     # loss = self._loss_f(me, y, self.tasks_centroids[i]).view(-1)
            # #     # tot_loss += loss
            # #
            # #     # a_probs = [torch.softmax(self.calculate_similarity(_e  *
            # #     #                                                    alphas[i],
            # #     #                                                    self.tasks_centroids[i]),
            # #     #                          -1)
            # #     #            for _e in ae]
            # #     #
            # #     # a_probs = torch.stack(a_probs)
            # #     #
            # #     # entropy = -(a_probs.log() * a_probs).sum(-1) / np.log(
            # #     #     a_probs.shape[-1])
            # #     # entropy = entropy.mean(0).cpu().numpy()
            # #     #
            # #     # entropy_distance.append(entropy - self.thresholds[i][-1])
            # #     # scores.append(entropy)
            # #
            # #     sim = self.calculate_similarity(me * alphas[i],
            # #                                     self.tasks_centroids[i])
            # #
            # #     loss = loss_f(sim)
            # #     tot_loss += loss
            # #
            # #     predictions.append(sim.argmax(1).cpu().numpy())
            # #
            # #     break
            # #
            # # # tot_loss = tot_loss.mean()
            # # # tot_loss.backward()
            # #
            # # grad = torch.autograd.grad(tot_loss, alphas, allow_unused=True)
            # #
            # # alphas = alphas - 1 * grad[0]
            #
            # # offset = offsets[correct_task]
            # # distances = np.stack(distances, 1)
            # # predictions = np.asarray(predictions)
            # # probs = np.asarray(probs)
            #
            # scores = np.asarray(scores)
            # predictions = np.asarray(predictions)
            #
            # # for i, (_, _, th) in enumerate(self.thresholds):
            # #     scores[i] -= th
            #
            # # predicted_tasks = scores.mean(-1).argmin(0)
            # predicted_tasks = scores.argmin(0)

            # correct_prediction_mask = predicted_tasks == correct_task
            # correct_prediction_mask = torch.tensor(correct_prediction_mask,
            #                                        dtype=torch.float,
            #                                        device=strategy.device)
            #
            # # pred = predictions[correct_task]
            #
            # # pred = self.custom_forward(strategy.model,
            # #                            strategy.mb_x,
            # #                            correct_task)
            #
            # pred = self._predict(strategy, correct_task)
            # pred = correct_prediction_mask * pred + (
            #         1 - correct_prediction_mask) * -1

            # pred = torch.tensor(pred,
            #                     device=strategy.device,
            #                     dtype=torch.long)

            # entropy_distance = np.asarray(entropy_distance)
            #
            # # probs = 1 - (distances / distances.sum(-1, keepdims=True))
            #
            # # counters = np.zeros_like(distances)
            # #
            # # for di, d in enumerate(distances):
            # #     for i in range(len(d)):
            # #         th = self.thresholds[i]
            # #         for j in range(d.shape[-1]):
            # #             counters[di, i, j] = (d[i, j] > th).sum()
            # #
            # # mx, mn = counters.max(-1), counters.min(-1)
            # #
            # # predicted_tasks = (mx - mn).argmax(-1)
            #
            # predicted_tasks = entropy_distance.argmin(0)
            # correct_prediction_mask = predicted_tasks == correct_task
            #
            # # pred = probs[:, correct_task].argmax(-1)
            # # pred += offsets[correct_task]
            #
            # pred = predictions[correct_task]
            # pred = correct_prediction_mask * pred + (
            #         1 - correct_prediction_mask) * -1
            #
            # # probs = np.exp(all_sims) / np.exp(all_sims).sum(-1, keepdims=True)
            # # entropy = -(probs * np.log(probs)).sum(-1) / np.log(2)
            # #
            # # predicted_task = distances.max(-1).argmax(1)[None, :]
            # #
            # # pred = np.take_along_axis(predictions, predicted_task, 0)[0]
            #
            # pred = torch.tensor(pred,
            #                     device=strategy.device,
            #                     dtype=torch.long)

        else:
            if len(self.tasks_centroids) == 0:
                centroids = self.current_centroids
            else:
                centroids = self.tasks_centroids[correct_task]

            if centroids is None:
                return torch.full_like(strategy.mb_y, -1)

            sim = self.calculate_similarity(embeddings, centroids)
            pred = torch.argmax(sim, 1)

        strategy.model.train(mode)

        return pred

    def before_backward(self, strategy, **kwargs):
        if len(self.tasks_centroids) > 0:

            x, y, t = strategy.mbatch
            dist = 0
            tot_sim = 0

            if self.sit:
                # maximize the entropy of the current task wrt other heads
                for i in range(len(self.tasks_centroids)):
                    dataloader = DataLoader(self.memory[i],
                                            batch_size=len(x),
                                            shuffle=True)

                    px, py, _ = next(iter(dataloader))
                    px = px.to(strategy.device)
                    py = py.to(strategy.device)

                    p_e = self.custom_forward(self.past_model, px, i,
                                              force_eval=True)

                    c_e = self.custom_forward(strategy.model, px, i,
                                              force_eval=True)

                    # sim = cosine_similarity(p_e, c_e, -1)
                    # _dist = 1 - sim

                    _dist = torch.norm(p_e - c_e, p=2, dim=1)
                    dist += _dist.mean()

                lens = [len(c) for c in self.tasks_centroids]
                offsets = np.cumsum(lens)

                offsets_tensor = torch.tensor([0] + offsets.tolist(),
                                              dtype=torch.long,
                                              device=strategy.device)

                concatenated_tasks = self.storage_policy.buffer

                loader = DataLoader(concatenated_tasks,
                                    batch_size=len(x),
                                    shuffle=True)

                past_x, past_y, past_t = next(iter(loader))
                past_x, past_y, past_t = past_x.to(strategy.device), past_y.to(strategy.device), past_t.to(strategy.device)
                x, y, t = torch.cat((x, past_x), 0), torch.cat((y, past_y)), torch.cat((t, past_t))

                e = 0

                y += torch.index_select(offsets_tensor, 0, t)

                for task in range(len(offsets_tensor)):
                    e += self.scaler(
                        self.custom_forward(strategy.model, x, task),
                        task)

                centroids = torch.cat([self.scaler(c, task)
                                       for task, c in
                                       enumerate(chain(self.tasks_centroids,
                                                       [self.current_centroids]))],
                                      0)

                loss = self._loss_f(e, y, centroids)
                loss = loss.view(-1).mean()

                if strategy.clock.train_epoch_iterations % 50 == 0:
                    print(loss)

                strategy.loss += loss

            else:
                for i in range(len(self.tasks_centroids)):
                    p_e = self.custom_forward(self.past_model, x, i,
                                              force_eval=True)

                    c_e = self.custom_forward(strategy.model, x, i,
                                              force_eval=True)

                    _dist = torch.norm(p_e - c_e, p=2, dim=1)
                    dist += _dist.mean()

            dist = dist / len(self.tasks_centroids)

            dist = dist * self.penalty_weight
            tot_sim = tot_sim * self.sit_penalty_wights

            if strategy.clock.train_epoch_iterations % 50 == 0:
                print(strategy.loss, dist, tot_sim)

            strategy.loss += (dist + tot_sim)

            # self.past_model = deepcopy(strategy.model)

# class _ContinualMetricLearningPlugin(StrategyPlugin):
#     def __init__(self, penalty_weight: float, sit=False,
#                  sit_penalty_wights: float = 0.1):
#
#         super().__init__()
#
#         self.past_model = None
#         self.penalty_weight = penalty_weight
#         self.sit_penalty_wights = sit_penalty_wights
#
#         self.similarity = 'euclidean'
#         self.tasks_centroids = []
#         self.sit = sit
#         self.tasks_forest = {}
#
#     def calculate_centroids(self, strategy: BaseStrategy, dataset):
#
#         # model = strategy.model
#         device = strategy.device
#         dataloader = DataLoader(dataset,
#                                 batch_size=strategy.train_mb_size)
#         # batch_size=len(dataset))
#
#         classes = set(dataset.targets)
#
#         embs = defaultdict(list)
#
#         # classes = set()
#         # for x, y, tid in data:
#         #     x, y, _ = d
#         #     emb, _ = strategy.model.forward_single_task(x, t, True)
#         #     classes.update(y.detach().cpu().tolist())
#
#         classes = sorted(classes)
#
#         for d in dataloader:
#             x, y, tid = d
#             x = x.to(device)
#             # embeddings = strategy.model.forward_single_task(x, tid, False)
#
#             embeddings = strategy.model(x, tid)
#             if self.sit and len(self.tasks_centroids) > 0:
#                 for i in range(len(self.tasks_centroids)):
#                     embeddings += strategy.model(x, i)
#
#             for c in classes:
#                 embs[c].append(embeddings[y == c])
#
#         embs = {c: torch.cat(e, 0) for c, e in embs.items()}
#
#         centroids = torch.stack([torch.mean(embs[c], 0) for c in classes], 0)
#
#         # if self.sit and len(self.tasks_centroids) > 0:
#         #     centroids_means = torch.cat(self.tasks_centroids, 0).sum(0,
#         #                                                              keepdims=True)
#         #     centroids += centroids_means
#
#         return centroids
#
#     def calculate_similarity(self, x, y, similarity: str = None, sigma=1):
#         if similarity is None:
#             similarity = self.similarity
#
#         n = x.size(0)
#         m = y.size(0)
#         d = x.size(1)
#         if d != y.size(1):
#             raise Exception
#
#         a = x.unsqueeze(1).expand(n, m, d)
#         b = y.unsqueeze(0).expand(n, m, d)
#
#         if similarity == 'euclidean':
#             dist = -torch.pow(a - b, 2).sum(2).sqrt()
#         elif similarity == 'rbf':
#             dist = -torch.pow(a - b, 2).sum(2).sqrt()
#             dist = dist / (2 * sigma ** 2)
#             dist = torch.exp(dist)
#         elif similarity == 'cosine':
#             sim = cosine_similarity(a, b, -1)
#             dist = (sim + 1) / 2
#         else:
#             assert False
#
#         return dist
#
#     def loss(self, strategy, **kwargs):
#         if not strategy.model.training:
#             return -1
#
#         centroids = self.calculate_centroids(strategy,
#                                              strategy.experience.dev_dataset)
#
#         mb_output, y, x = strategy.mb_output, strategy.mb_y, strategy.mb_x
#
#         # if self.sit and len(self.tasks_centroids) > 0:
#         #     for i in range(len(self.tasks_centroids)):
#         #         mb_output += strategy.model(x, i)
#
#         sim = self.calculate_similarity(mb_output, centroids)
#
#         log_p_y = log_softmax(sim, dim=1)
#         loss_val = -log_p_y.gather(1, y.unsqueeze(-1))
#         loss_val = loss_val.view(-1).mean()
#
#         return loss_val
#
#     @torch.no_grad()
#     def after_training_exp(self, strategy, **kwargs):
#         tid = strategy.experience.current_experience
#
#         centroids = self.calculate_centroids(strategy,
#                                              strategy.experience.
#                                              dev_dataset)
#
#         self.tasks_centroids.append(centroids)
#
#         self.past_model = deepcopy(strategy.model)
#         self.past_model.eval()
#
#         if isinstance(strategy.model, BatchNormModelWrap):
#             for name, module in strategy.model.named_modules():
#                 if isinstance(module, TaskIncrementalBatchNorm2D):
#                     module.freeze_eval(tid)
#                     for p in module.parameters():
#                         p.requires_grad_(False)
#
#         # for param in strategy.model.classifier.classifiers[str(tid)].parameters():
#         #     param.requires_grad_(False)
#
#         if self.sit:
#             dataloader = DataLoader(strategy.experience.dev_dataset,
#                                     batch_size=strategy.train_mb_size)
#
#             embeddings = []
#             for x, _, t in dataloader:
#                 x = x.to(strategy.device)
#
#                 # e = strategy.model.forward_single_task(x, tid, False)
#                 e = self.past_model(x, t)
#
#                 if self.sit and len(self.tasks_centroids) > 0:
#                     for i in range(tid):
#                         e += self.past_model(x, i)
#
#                 embeddings.extend(e.cpu().tolist())
#
#             embeddings = np.asarray(embeddings)
#
#             ln = len(embeddings)
#             nn = 50 if ln > 100 else ln // 3
#
#             # f = LocalOutlierFactor(novelty=True,
#             #                        n_neighbors=50,
#             #                        contamination=0.1)
#
#             # f = OneClassSVM(nu=0.1, kernel='linear')
#
#             f = IsolationForest(n_estimators=500,
#                                 n_jobs=-1,
#                                 contamination=0.1,
#                                 max_samples=0.7)
#
#             f.fit(embeddings)
#
#             self.tasks_forest[tid] = f
#
#     @torch.no_grad()
#     def calculate_classes(self, strategy, embeddings):
#
#         mode = strategy.model.training
#         strategy.model.eval()
#
#         correct_task = strategy.experience.current_experience
#
#         if False and self.sit and len(self.tasks_centroids) > 1:
#
#             correct_embs = None
#             ood_predictions = []
#
#             ssims = []
#             _ssims = []
#             predictions = []
#
#             for ti, forest in self.tasks_forest.items():
#                 e = strategy.model(strategy.mb_x, ti)
#
#                 if self.sit and len(self.tasks_centroids) > 0:
#                     for i in range(ti):
#                         e += strategy.model(strategy.mb_x, i)
#
#                 sim = self.calculate_similarity(e, self.tasks_centroids[ti])
#                 _ssims.append(sim.cpu().numpy())
#
#                 sim = softmax(sim, -1).cpu().numpy()
#
#                 ssims.append(sim.max(1))
#                 predictions.append(sim.argmax(1))
#
#                 if ti == correct_task:
#                     correct_embs = e
#
#                 e = e.cpu().numpy()
#
#                 forest_prediction = forest.decision_function(e)
#                 # forest_prediction = forest.score_samples(e)
#
#                 # mn, std = forest_prediction.mean(), forest_prediction.std()
#                 #
#                 # nv = forest_prediction - mn
#                 # nv = nv / (2**0.5 * std)
#                 # forest_prediction = erf(nv)
#
#                 # forest_prediction = np.maximum(0, nv)
#
#                 ood_predictions.append(forest_prediction)
#
#             ood_predictions = np.asarray(ood_predictions)
#             # ood_predictions = ood_predictions + 0.5
#
#             ssims = np.asarray(ssims)
#             _ssims = np.asarray(_ssims)
#             predictions = np.asarray(predictions)
#
#             # mn, std = predictions.mean(1, keepdims=True), \
#             #           predictions.std(1, keepdims=True)
#             #
#             # nv = predictions - mn
#             # nv = nv / ((2 ** 0.5) * std)
#             # predictions = erf(nv)
#
#             predicted_tasks = np.argmax(ood_predictions, 0)
#             # predicted_tasks = np.argmax(ssims, 0)
#
#             correct_prediction_mask = predicted_tasks == correct_task
#
#             pred = correct_prediction_mask * predictions[correct_task] + (
#                     1 - correct_prediction_mask) * -1
#
#             pred = torch.tensor(pred,
#                                 device=strategy.device,
#                                 dtype=torch.long)
#
#             # correct_prediction_mask = torch.tensor(correct_prediction_mask,
#             #                                        device=strategy.device,
#             #                                        dtype=torch.long)
#             #
#             # centroids = self.tasks_centroids[correct_task]
#             # sim = self.calculate_similarity(correct_embs, centroids)
#             # sm = softmax(sim, -1)
#             # sm = torch.argmax(sm, 1)
#             #
#             # pred = correct_prediction_mask * sm + (
#             #         1 - correct_prediction_mask) * -1
#
#         else:
#
#             sim = self.calculate_similarity(embeddings,
#                                             self.tasks_centroids[correct_task])
#             sm = softmax(sim, -1)
#             pred = torch.argmax(sm, 1)
#
#         strategy.model.train(mode)
#
#         return pred
#
#     def before_backward(self, strategy, **kwargs):
#         if strategy.clock.train_exp_counter > 0:
#
#             # strategy.model.eval()
#
#             x, _, tdi = strategy.mbatch
#             dist = 0
#             tot_sim = 0
#
#             # # if self.sit:
#             # dataloader = DataLoader(strategy.experience.dev_dataset,
#             #                         batch_size=len(strategy.experience.dev_dataset),
#             #                         shuffle=True)
#             #
#             # dev_x, _, dev_t = next(iter(dataloader))
#             #
#             # dev_x, dev_t = dev_x.to(strategy.device), dev_t.to(
#             #     strategy.device)
#
#             for i in range(len(self.tasks_centroids)):
#                 p_e = self.past_model(x, i)
#                 c_e = strategy.model(x, i)
#
#                 # if False:
#                 #     p_e = normalize(p_e)
#                 #     c_e = normalize(c_e)
#
#                 _dist = torch.norm(p_e - c_e, p=2, dim=1)
#                 dist += _dist.mean()
#
#                 if self.sit:
#                     e = strategy.model(x, i)
#
#                     if self.sit and len(self.tasks_centroids) > 0:
#                         for j in range(i):
#                             e += strategy.model(x, j)
#
#                     d = -self.calculate_similarity(e, self.tasks_centroids[i])
#
#                     sim = 1 / (1 + d)
#
#                     tot_sim += sim.mean(1).mean(0)
#
#             # tot_sim = tot_sim / len(self.tasks_centroids)
#             # dist = dist / len(self.tasks_centroids)
#
#             dist = dist * self.penalty_weight
#             tot_sim = tot_sim * self.sit_penalty_wights
#
#             if strategy.clock.train_exp_counter > 0:
#                 print(tot_sim, dist, strategy.loss)
#
#             strategy.loss += dist + tot_sim
#
#             # strategy.model.train()

# class ClassIncrementalContinualMetricLearningPlugin(StrategyPlugin):
#     def __init__(self, penalty_weight: float, sit=False):
#         super().__init__()
#
#         self.past_model = None
#         self.penalty_weight = penalty_weight
#         self.similarity = 'euclidean'
#
#         self.tasks_centroids = []
#         self.concatenated_centroids = None
#         self.support_sets = []
#
#         self.sit = sit
#         self.tasks_forest = {}
#
#     @staticmethod
#     def calculate_centroids(strategy: BaseStrategy, dataset):
#
#         training = strategy.model.training
#         strategy.model.eval()
#
#         device = strategy.device
#         dataloader = DataLoader(dataset,
#                                 batch_size=strategy.train_mb_size)
#
#         classes = set(dataset.targets)
#
#         # mask = torch.zeros(max(classes) + 1, dtype=torch.float, device=device)
#         # for c in classes:
#         #     mask[c] = 1
#
#         embs = defaultdict(list)
#
#         classes = sorted(classes)
#
#         for d in dataloader:
#             x, y, tid = d
#             x = x.to(device)
#             embeddings = strategy.model(x, tid)
#             for c in classes:
#                 embs[c].append(embeddings[y == c])
#
#         embs = {c: torch.cat(e, 0) for c, e in embs.items()}
#         # zeros = torch.zeros_like(embs[classes[0]][0])
#         # centroids = torch.stack([torch.mean(embs[c], 0)
#         #                          if c in classes else zeros
#         #                          for c in range(max(classes))], 0)
#         centroids = torch.stack([torch.mean(embs[c], 0)
#                                  for c in classes], 0)
#
#         strategy.model.train(training)
#
#         return centroids
#
#     def calculate_similarity(self, x, y, similarity: str = None, sigma=1):
#         if similarity is None:
#             similarity = self.similarity
#
#         n = x.size(0)
#         m = y.size(0)
#         d = x.size(1)
#         if d != y.size(1):
#             raise Exception
#
#         a = x.unsqueeze(1).expand(n, m, d)
#         b = y.unsqueeze(0).expand(n, m, d)
#
#         if similarity == 'euclidean':
#             dist = -torch.pow(a - b, 2).sum(2).sqrt()
#         elif similarity == 'rbf':
#             dist = -torch.pow(a - b, 2).sum(2).sqrt()
#             dist = dist / (2 * sigma ** 2)
#             dist = torch.exp(dist)
#         elif similarity == 'cosine':
#             dist = cosine_similarity(a, b, -1) ** 2
#         else:
#             assert False
#
#         return dist
#
#     def loss(self, strategy, **kwargs):
#
#         if not strategy.model.training:
#             return 0
#
#         centroids = self.calculate_centroids(strategy,
#                                              strategy.experience.dev_dataset)
#
#         if len(self.tasks_centroids) > 0:
#             centroids = torch.cat((self.concatenated_centroids, centroids), 0)
#
#         mb_output, y = strategy.mb_output, strategy.mb_y
#
#         sim = self.calculate_similarity(mb_output, centroids)
#
#         log_p_y = log_softmax(sim, dim=1)
#         loss_val = -log_p_y.gather(1, y.unsqueeze(-1))
#         loss_val = loss_val.view(-1).mean()
#
#         return loss_val
#
#     @torch.no_grad()
#     def after_training_exp(self, strategy, **kwargs):
#         tid = strategy.experience.current_experience
#
#         centroids = self.calculate_centroids(strategy,
#                                              strategy.experience.
#                                              dev_dataset)
#
#         self.tasks_centroids.append(centroids)
#         self.support_sets.append(strategy.experience.dev_dataset)
#
#         self.concatenated_centroids = torch.cat(self.tasks_centroids, 0)
#
#     @torch.no_grad()
#     def calculate_classes(self, strategy, embeddings):
#
#         centroids = torch.cat(self.tasks_centroids, 0)
#         if len(self.tasks_centroids) > 0:
#             centroids = torch.cat((self.concatenated_centroids, centroids), 0)
#
#         sim = self.calculate_similarity(embeddings, centroids)
#         sm = softmax(sim, -1)
#         pred = torch.argmax(sm, 1)
#
#         return pred
#
#     def before_update(self, strategy, **kwargs):
#         if strategy.clock.train_exp_counter > 0:
#
#             # strategy.loss = 0
#             # strategy.optimizer.zero_grad()
#
#             past_model = deepcopy(strategy.model)
#             past_model.zero_grad()
#
#             strategy.optimizer.step()
#             strategy.optimizer.zero_grad()
#
#             l = strategy.loss
#
#             # l = strategy._criterion(
#             #     past_model.model(strategy.mb_x, strategy.mb_task_id),
#             #     strategy.mb_y)
#
#             # strategy.model.eval()
#
#             # a = strategy.model.model.backbone.model.bn1.bns['0'].weight
#             # b = self.past_model.model.backbone.model.bn1.bns['0'].weight
#
#             # x, _, tdi = strategy.mbatch
#             dist = 0
#             # sim = 0
#
#             # ce = strategy.model.forward(x, strategy.clock.train_exp_counter)
#             for tid, d in enumerate(self.support_sets):
#                 nc = self.calculate_centroids(strategy, d)
#                 # nc = nc + (torch.randn_like(nc) * 0.1)
#                 # for c, _c in zip(self.tasks_centroids, nc):
#                 c = self.tasks_centroids[tid]
#                 _dist = torch.norm(nc - c, p=2, dim=1)
#                 dist += _dist.sum()
#
#             # dist = dist / self.concatenated_centroids.shape[0]
#
#             # for i in range(len(self.tasks_centroids)):
#             #     p_e = self.past_model.forward(x, i)
#             #     c_e = strategy.model.forward(x, i)
#             #
#             #     if False:
#             #         p_e = normalize(p_e)
#             #         c_e = normalize(c_e)
#             #
#             #     _dist = torch.norm(p_e - c_e, p=2, dim=1)
#             #     dist += _dist.mean()
#             #
#             #     # if self.sit and i != strategy.clock.train_exp_counter:
#             #     #     pe = strategy.model.forward(x, i)
#             #     #
#             #     #     sim += (1 / (1 + torch.norm(ce - pe, p=2, dim=1))).mean()
#             #     #     # print(sim)
#
#             dist = dist * self.penalty_weight
#
#             # print(dist, strategy.loss)
#
#             # strategy.loss += dist
#
#             l = l + dist
#
#             l.backward()
#
#             grads = {name: p.grad
#                      for name, p in strategy.model.named_parameters()}
#
#             strategy.model.load_state_dict(self.past_model.state_dict())
#
#             for name, p in strategy.model.named_parameters():
#                 p.grad = grads[name]
#
#             # strategy.loss.backward()
#             # strategy.optimizer.step()
#
#             # strategy.model.train()
#
#             # tdi = strategy.experience.current_experience
#             # if self.sit:
#             #     tot_sim = 0
#             #     p_e = strategy.model.forward_single_task(x, tdi)
#             #
#             #     for i in range(tdi):
#             #         # for j in range(i + 1, len(self.tasks_centroids)):
#             #         #     p_e = strategy.model.forward_single_task(x, i)
#             #         c_e = strategy.model.forward_single_task(x, i)
#             #
#             #         _dist = torch.norm(p_e - c_e, p=2, dim=1)
#             #         sim = 1 / (1 + _dist)
#             #
#             #         tot_sim += sim.mean()
#             #
#             #     strategy.loss += tot_sim * self.penalty_weight
