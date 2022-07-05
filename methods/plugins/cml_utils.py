from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import chain

import numpy as np
import torch
from avalanche.benchmarks.utils import AvalancheSubset, AvalancheConcatDataset, \
    AvalancheDataset
from avalanche.models import MultiTaskModule
from avalanche.training import BalancedExemplarsBuffer, BaseStrategy, \
    ReservoirSamplingBuffer
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from torch import nn, softmax, autograd, log_softmax
from torch.nn import BatchNorm2d
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torchvision.transforms.functional as F


class CustomExperienceBalancedBuffer(BalancedExemplarsBuffer):
    """ Rehearsal buffer with samples balanced over experiences.

    The number of experiences can be fixed up front or adaptive, based on
    the 'adaptive_size' attribute. When adaptive, the memory is equally
    divided over all the unique observed experiences so far.
    """

    def __init__(self, max_size: int, adaptive_size: bool = True,
                 num_experiences=None):
        """
        :param max_size: max number of total input samples in the replay
            memory.
        :param adaptive_size: True if mem_size is divided equally over all
                              observed experiences (keys in replay_mem).
        :param num_experiences: If adaptive size is False, the fixed number
                                of experiences to divide capacity over.
        """
        super().__init__(max_size, adaptive_size, num_experiences)

    def update(self, dataset, strategy: "BaseStrategy", **kwargs):
        new_data = dataset
        num_exps = strategy.clock.train_exp_counter + 1
        lens = self.get_group_lengths(num_exps)

        new_buffer = ReservoirSamplingBuffer(lens[-1])
        new_buffer.update_from_dataset(new_data)
        self.buffer_groups[num_exps - 1] = new_buffer

        for ll, b in zip(lens, self.buffer_groups.values()):
            b.resize(strategy, ll)


class Memory(ABC):
    def __init__(self, memory_size, adaptive_size: bool = True, **kwargs):

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
            lengths = [self.memory_size // len(self._memory) for _ in
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
    def add_task(self, dataset, tid, model, **kwargs):
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

                        # weights = [w / sum(scores[y]) for w in scores[y]]
                        # i = np.random.choice(np.arange(len(weights)),
                        #                      p=weights)
                        # r = np.random.uniform()
                        #
                        # # più è distante e meno è probabile che venga cambiato
                        # if r < scores[y][i] / (scores[y][i] + max_distance):

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
        # self.samples_per_centroid = samples_per_centroid
        self.n_clusters = n_clusters

        # self.random_sample = random_sample

        if cluster_type == 'kmeans':
            self._algo = KMeans
        elif cluster_type == 'spectral':
            self._algo = SpectralClustering
        else:
            assert False

    @torch.no_grad()
    def add_task(self, dataset, tid, model, **kwargs):
        device = next(model.parameters()).device

        embs = []
        ys = []

        for index, (x, y, t) in enumerate(DataLoader(dataset, 32)):

            x = x.to(device)

            o = model(x, tid)
            o = o.cpu().numpy()

            embs.append(o)
            ys.extend(y.tolist())

        embs = np.concatenate(embs, 0)
        ys = np.array(ys)
        indexes = np.arange(len(ys))

        lens = self.get_group_lengths(len(indexes) + len(self.buffer_datasets))

        for ln, y in zip(lens[:len(indexes)], np.unique(ys)):
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
                _i = _i[:ln]

                idx_per_cluster[l] = _i

            all_indexes = list(chain(idx_per_cluster.items()))

            val = len(all_indexes)
            val1 = ln * len(self.n_clusters)

            if val < val1:
                np.random.shuffle(indexes)

                for i in indexes:
                    if i not in all_indexes:
                        all_indexes.append(i)

                    if len(all_indexes) == val1:
                        break

            new_data = AvalancheSubset(dataset, all_indexes)

            new_buffer = ReservoirSamplingBuffer(lens[-1])
            new_buffer.update_from_dataset(new_data)

            self._memory[(tid, y)] = new_buffer

        for ll, b in zip(lens, self._memory.values()):
            b.resize(None, ll)


class _ClusteringMemory:
    def __init__(self,
                 samples_per_centroid,
                 n_clusters,
                 cluster_type='kmeans',
                 random_sample=True, **kwargs):

        self.memory = {}
        self.samples_per_centroid = samples_per_centroid
        self.n_clusters = n_clusters

        self.random_sample = random_sample

        if cluster_type == 'kmeans':
            self._algo = KMeans
        elif cluster_type == 'spectral':
            self._algo = SpectralClustering
        else:
            assert False

    @torch.no_grad()
    def add_task(self, dataset, tid, model, **kwargs):
        device = next(model.parameters()).device

        embs = []
        ys = []

        for index, (x, y, t) in enumerate(DataLoader(dataset, 32)):

            x = x.to(device)

            o = model(x, tid)
            o = o.cpu().numpy()

            embs.append(o)
            ys.extend(y.tolist())

        embs = np.concatenate(embs, 0)
        ys = np.array(ys)
        indexes = np.arange(len(ys))

        for y in np.unique(ys):
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
                _i = _i[:self.samples_per_centroid]

                idx_per_cluster[l] = _i

                # self.memory[(tid, y, l)] = AvalancheSubset(dataset, _i)

            all_indexes = list(chain(idx_per_cluster.items()))

            val = len(all_indexes)
            val1 = self.samples_per_centroid * len(self.n_clusters)

            if val < val1:
                np.random.shuffle(indexes)

                for i in indexes:
                    if i not in all_indexes:
                        all_indexes.append(i)

                    if len(all_indexes) == val1:
                        break

            self.memory[(tid, y)] = AvalancheSubset(dataset, all_indexes)

            # distances = km.transform(_embs)
            # for i in range(self.n_clusters):
            #     d = distances[:, i]
            #     ind = np.argsort(d)[:self.samples_per_centroid]
            #
            #     self.memory[(tid, y, i)] = AvalancheSubset(dataset, indexes[ind])

        # for k in list(indexes):
        #     self.memory[(tid, k)] = AvalancheSubset(dataset, indexes[k])

    def concatenated_dataset(self):
        return AvalancheConcatDataset(self.memory.values())


class RandomMemory(Memory):
    def __init__(self, samples_per_class, memory_size, **kwargs):
        super().__init__(memory_size, **kwargs)

    @torch.no_grad()
    def add_task(self, dataset, tid, **kwargs):
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

        for ll, b in zip(lens, self._memory.values()):
            b.resize(None, ll)

    def concatenated_dataset(self):
        return AvalancheConcatDataset(self._memory.values())


class _RandomMemory:
    def __init__(self, samples_per_class, **kwargs):
        self.memory = {}
        self.samples_per_class = samples_per_class

    @torch.no_grad()
    def add_task(self, dataset, tid, **kwargs):
        indexes = defaultdict(list)

        for index, (x, y, t) in enumerate(DataLoader(dataset, 1)):

            indexes[y.item()].append(index)

        for k in list(indexes):
            i = np.asarray(indexes[k])
            np.random.shuffle(i)

            self.memory[(tid, k)] = AvalancheSubset(dataset,
                                                    i[:self.samples_per_class])

    def concatenated_dataset(self):
        return AvalancheConcatDataset(self.memory.values())


class GeneratedMemory:
    def __init__(self, samples_per_class, **kwargs):
        self.memory = {}
        self.samples_per_class = samples_per_class

        self.lr = 0.1
        self.iterations = 100

    def calculate_similarity(self, x, y, distance: str = None, sigma=1):
        # if distance is None:
        #     distance = self.similarity

        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        if d != y.size(1):
            raise Exception

        a = x.unsqueeze(1).expand(n, m, d)
        b = y.unsqueeze(0).expand(n, m, d)

        # if distance == 'euclidean':
        similarity = -torch.pow(a - b, 2).sum(2).sqrt()
        # elif distance == 'rbf':
        #     similarity = -torch.pow(a - b, 2).sum(2).sqrt()
        #     similarity = similarity / (2 * sigma ** 2)
        #     similarity = torch.exp(similarity)
        # elif distance == 'cosine':
        #     similarity = cosine_similarity(a, b, -1)
        # else:
        #     assert False

        return similarity

    def add_task(self,
                 dataset: AvalancheDataset,
                 tid: int,
                 model: nn.Module,
                 **kwargs):

        def show(imgs):
            if not isinstance(imgs, list):
                imgs = [imgs]
            fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
            for i, img in enumerate(imgs):
                img = img.detach()
                img = F.to_pil_image(img)
                axs[0, i].imshow(np.asarray(img))
                axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[],
                              yticks=[])
            plt.show()
            plt.close(fig)

        device = next(model.parameters()).device

        classes = np.unique(dataset.targets)
        x, _, _ = next(iter(DataLoader(dataset, 32)))
        mn, mx = x.min(), x.max()

        shape = (self.samples_per_class, ) + x.shape[1:]

        for current_class in classes:
            synthetic_images = (mn - mx) * torch.rand(*shape, device=device, requires_grad=True) + mx
            synthetic_images.to(device)

            for i in range(self.iterations):
                for index, (x, y, t) in enumerate(DataLoader(dataset,
                                                             32,
                                                             shuffle=True)):

                    x, y, t = x.to(device), y.to(device), t.to(device)

                    os = model(synthetic_images, tid)
                    o = model(x, tid)

                    centroids = torch.stack([(o[y == c]).mean(0)
                                             for c in classes], 0)

                    sim = self.calculate_similarity(os, centroids)
                    loss = -log_softmax(sim, dim=1).gather(1, torch.full_like(y, current_class).unsqueeze(-1))
                    loss = loss.view(-1).mean()

                    grads = autograd.grad(loss, synthetic_images, retain_graph=False, create_graph=False)[0]

                    synthetic_images = synthetic_images - self.lr * grads
                    synthetic_images = torch.clamp(synthetic_images, mn, mx)
                    synthetic_images = synthetic_images.cpu().to(device)

                print(loss)

                grid = make_grid(synthetic_images.cpu())

                show(grid)

        #     indexes[y.item()].append(index)
        #
        # for k in list(indexes):
        #     i = np.asarray(indexes[k])
        #     np.random.shuffle(i)
        #
        #     self.memory[(tid, k)] = AvalancheSubset(dataset,
        #                                             i[:self.samples_per_class])

    def concatenated_dataset(self):
        return AvalancheConcatDataset(self.memory.values())


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

    def add_task(self, embedding_size):
        if self.proj_type == 'offset':
            self.values.append(nn.Parameter(torch.randn(embedding_size)))

        # elif self.proj_type == 'embeddings':
        #     emb = nn.Embedding(num_embeddings=n_tasks,
        #                        embedding_dim=embedding_size)
        #     linear = nn.Sequential(nn.Linear(embedding_size, embedding_size),
        #                            nn.ReLU(),
        #                            nn.Linear(embedding_size, embedding_size))
        #     params = nn.ModuleList([emb, linear])
        #
        #     self.values = params
        elif self.proj_type == 'mlp':
            p = nn.Sequential(
                nn.Linear(embedding_size, embedding_size),
                nn.ReLU(),
                nn.Linear(embedding_size, embedding_size))

            self.values.append(p)

    # def reset(self, n_tasks, embedding_size):
    #     if self.proj_type == 'offset':
    #         offsets = nn.ParameterList(
    #             [nn.Parameter(torch.randn(embedding_size))
    #              for _ in range(n_tasks)])
    #
    #         self.values = offsets
    #
    #     elif self.proj_type == 'embeddings':
    #         emb = nn.Embedding(num_embeddings=n_tasks,
    #                            embedding_dim=embedding_size)
    #         linear = nn.Sequential(nn.Linear(embedding_size, embedding_size),
    #                                nn.ReLU(),
    #                                nn.Linear(embedding_size, embedding_size))
    #         params = nn.ModuleList([emb, linear])
    #
    #         self.values = params
    #     elif self.proj_type == 'mlp':
    #         params = nn.ModuleList([nn.Sequential(
    #             nn.Linear(embedding_size, embedding_size),
    #             nn.ReLU(),
    #             nn.Linear(embedding_size, embedding_size))
    #             for _ in range(n_tasks)])
    #
    #         self.values = params

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
        s = nn.Sequential(nn.Linear(embedding_size, embedding_size),
                          nn.ReLU(),
                          nn.Linear(embedding_size, embedding_size),
                          nn.Sigmoid())

        t = nn.Sequential(nn.Linear(embedding_size, embedding_size),
                          nn.ReLU(),
                          nn.Linear(embedding_size, embedding_size))

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
