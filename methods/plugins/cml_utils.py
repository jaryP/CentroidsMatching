from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
import torch
from avalanche.benchmarks.utils import AvalancheSubset, AvalancheConcatDataset, \
    AvalancheDataset
from avalanche.models import MultiTaskModule
from avalanche.training import BalancedExemplarsBuffer, BaseStrategy, \
    ReservoirSamplingBuffer
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from torch import nn, autograd, log_softmax
from torch.nn import BatchNorm2d
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torchvision.transforms.functional as F


class Memory(ABC):
    def __init__(self, memory_size,
                 adaptive_size: bool = True,
                 **kwargs):

        self.memory_size = memory_size
        self.adaptive_size = adaptive_size

        self._memory = {}

    def get_group_lengths(self, num_groups):
        if self.adaptive_size:
            lengths = [self.memory_size // num_groups for _ in range(num_groups)]
            rem = self.memory_size - sum(lengths)
            for i in range(rem):
                lengths[i] += 1
        else:
            lengths = [self.memory_size for _ in
                       range(num_groups)]
        return lengths

    def resize(self, new_size):
        """ Update the maximum size of the buffers. """

        self.memory_size = new_size

        lens = self.get_group_lengths(len(self._memory))
        for ll, buffer in zip(lens, self._memory.values()):
            buffer.resize(ll)

    @property
    def buffer_datasets(self):
        return [g.buffer for g in self._memory.values()]

    @property
    def buffer(self):
        return AvalancheConcatDataset(self.buffer_datasets)

    @abstractmethod
    def update(self, dataset, tid, model, **kwargs):
        pass


class DistanceMemory:
    def __init__(self, samples_per_class, **kwargs):
        self.memory = {}
        self.samples_per_class = samples_per_class

    def add_task(self, dataset, tid, model, **kwargs):
        device = next(model.parameters()).device

        indexes = defaultdict(list)
        embs = defaultdict(list)
        scores = defaultdict(list)

        with torch.no_grad():
            for index, (x, y, t) in enumerate(DataLoader(dataset, 1)):

                x, y = x.to(device), y.to(device)

                o = model(x, tid)
                o = o.cpu().numpy()[0]

                y = y.item()

                if len(embs[y]) == 0:
                    indexes[y].append(index)
                    scores[y].append(0)
                    embs[y].append(o)
                else:

                    if len(embs[y]) < self.samples_per_class // 3:
                        subset = embs[y]
                    else:
                        subset = np.random.choice(np.arange(len(embs[y])),
                                                  self.samples_per_class // 3,
                                                  False)
                        subset = [embs[y][c] for c in subset]

                    euclidean_similarity = [
                        1 / (1 + np.linalg.norm(o / np.linalg.norm(o) -
                                                c / np.linalg.norm(c))) for
                        c in subset]

                    max_distance = np.mean(euclidean_similarity)

                    if len(embs[y]) < self.samples_per_class:
                        indexes[y].append(index)
                        scores[y].append(max_distance)
                        embs[y].append(o)

                    elif max_distance < 0.5:

                        ss = np.asarray(scores[y])
                        ss = ss / (ss + max_distance)
                        weights = ss / ss.sum()

                        i = np.random.choice(np.arange(len(weights)),
                                             p=weights)

                        indexes[y][i] = index
                        scores[y][i] = max_distance
                        embs[y][i] = o

        for k in list(indexes):
            self.memory[(tid, k)] = AvalancheSubset(dataset, indexes[k])

    def concatenated_dataset(self):
        return AvalancheConcatDataset(self.memory.values())


class ClusteringMemory(Memory):
    def __init__(self,
                 memory_size,
                 n_clusters,
                 cluster_type='kmeans',
                 **kwargs):

        super().__init__(memory_size, **kwargs)

        self.memory = {}
        self.n_clusters = n_clusters

        if cluster_type == 'kmeans':
            self._algo = KMeans
        elif cluster_type == 'spectral':
            self._algo = SpectralClustering
        else:
            assert False

    @torch.no_grad()
    def update(self, dataset, tid, model, **kwargs):
        device = next(model.parameters()).device
        mode = model.training
        model.eval()

        embs = []
        ys = []

        for index, (x, y, t) in enumerate(DataLoader(dataset, 32)):

            x = x.to(device)

            o = model(x, tid)
            o = o.cpu().numpy()

            embs.append(o)
            ys.extend(y.tolist())

        model.train(mode)

        embs = np.concatenate(embs, 0)
        ys = np.array(ys)
        indexes = np.arange(len(ys))
        classes = np.unique(ys)

        lens = self.get_group_lengths(len(classes) + len(self.buffer_datasets))

        for ln, y in zip(lens[:len(classes)], classes):

            c_ln = round(ln / self.n_clusters)

            mask = ys == y

            _embs = embs[mask]
            _indexes = indexes[mask]

            algo = self._algo(n_clusters=self.n_clusters)
            algo.fit(_embs)

            labels = algo.labels_

            idx_per_cluster = {}

            for l in np.unique(labels):
                _i = _indexes[labels == l]
                np.random.shuffle(_i)
                _i = _i[:c_ln]

                idx_per_cluster[l] = _i

            all_indexes = list(np.concatenate(list(idx_per_cluster.values())))

            val = len(all_indexes)

            if val < c_ln:
                np.random.shuffle(indexes)

                for i in indexes:
                    if i not in all_indexes:
                        all_indexes.append(i)

                    if len(all_indexes) == c_ln:
                        break

            new_data = AvalancheSubset(dataset, all_indexes)

            new_buffer = ReservoirSamplingBuffer(lens[-1])
            new_buffer.update_from_dataset(new_data)

            self._memory[(tid, y)] = new_buffer

        if self.adaptive_size:
            for ll, b in zip(lens, self._memory.values()):
                b.resize(None, ll)


class RandomMemory(Memory):
    def __init__(self, memory_size, **kwargs):
        super().__init__(memory_size, **kwargs)

    @torch.no_grad()
    def update(self, dataset, tid, **kwargs):
        indexes = defaultdict(list)

        for index, (x, y, t) in enumerate(DataLoader(dataset, 1)):

            indexes[y.item()].append(index)

        lens = self.get_group_lengths(len(indexes) + len(self.buffer_datasets))

        for j, k in enumerate(indexes):
            i = np.asarray(indexes[k])
            np.random.shuffle(i)

            new_data = AvalancheSubset(dataset, i[:lens[j]])

            new_buffer = ReservoirSamplingBuffer(lens[-1])
            new_buffer.update_from_dataset(new_data)

            self._memory[(tid, k)] = new_buffer

        if self.adaptive_size:
            for ll, b in zip(lens, self._memory.values()):
                b.resize(None, ll)

    def concatenated_dataset(self):
        return AvalancheConcatDataset(self._memory.values())


class FakeMerging(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def add_task(self, **kwargs):
        pass

    def forward(self, x, i, **kwargs):
        return x


class Projector(nn.Module):
    def __init__(self, proj_type='offset', device=None):
        super().__init__()

        self.proj_type = proj_type
        self.device = device

        if proj_type == 'offset':
            self.values = nn.ParameterList()
        elif proj_type == 'mlp':
            self.values = nn.ModuleList()
        else:
            assert False, 'Projection type must be one of: [offset, mlp]'

    def reset(self):
        if isinstance(self.values, nn.ParameterList):
            self.values = nn.ParameterList()
        else:
            self.values = nn.ModuleList()

    def add_task(self, embedding_size, out_size=None):
        if self.proj_type == 'offset':
            self.values.append(nn.Parameter(torch.randn(embedding_size)))

        elif self.proj_type == 'mlp':
            if out_size is None:
                out_size = embedding_size

            p = nn.Sequential(
                nn.ReLU(),
                nn.Linear(embedding_size, out_size))

            self.values.append(p)

    def forward(self, x, i):
        if self.proj_type == 'offset':
            return x + self.values[i]
        elif self.proj_type == 'embeddings':
            off = self.values[0](torch.tensor(i, dtype=torch.long))
            off = self.values[1](off)
            return x + off
        elif self.proj_type == 'mlp':
            return self.values[i](x)


class ScaleTranslate(nn.Module):
    def __init__(self, device=None):
        super().__init__()

        self.device = device

        self.s = nn.ModuleList()
        self.t = nn.ModuleList()

    def reset(self):
        self.s = nn.ModuleList()
        self.t = nn.ModuleList()

    def add_task(self, embedding_size):
        s = nn.Sequential(nn.ReLU(),
                          nn.Linear(embedding_size, embedding_size),
                          nn.Sigmoid())

        t = nn.Sequential(nn.ReLU(),
                          nn.Linear(embedding_size, embedding_size),
                          )

        self.s.append(s)
        self.t.append(t)

    def forward(self, x, i):
        return x * self.s[i](x) + self.t[i](x)


def wrap_model(model: nn.Module):
    for name, module in model.named_children():
        if isinstance(module, BatchNorm2d):
            setattr(model, name, TaskIncrementalBatchNorm2D(module))
        else:
            wrap_model(module)


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
