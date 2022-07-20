from collections import defaultdict
from copy import deepcopy
from itertools import chain

import numpy as np
import torch
from avalanche.benchmarks.utils import AvalancheConcatDataset, AvalancheSubset
from avalanche.benchmarks.utils.data_loader import GroupBalancedDataLoader, \
    ReplayDataLoader, TaskBalancedDataLoader
from avalanche.models import avalanche_forward
from avalanche.training import BaseStrategy, ExperienceBalancedBuffer
from avalanche.training.plugins import StrategyPlugin
from torch import cosine_similarity, log_softmax
from torch.optim import Adam
from torch.utils.data import DataLoader
import os

from methods.plugins.cml_utils import DistanceMemory, ClusteringMemory, \
    Projector, \
    ScaleTranslate, TaskIncrementalBatchNorm2D, BatchNormModelWrap, \
    RandomMemory, FakeMerging


class CentroidsMatching(StrategyPlugin):
    def __init__(self,
                 penalty_weight: float,
                 sit=False,
                 sit_memory_size: int = 200,
                 memory_type='random',
                 memory_parameters=None,
                 merging_strategy='scale_translate',
                 centroids_merging_strategy=None,
                 **kwargs):

        super().__init__()

        self.tasks_centroids = []

        self.past_model = None
        self.current_centroids = None
        self.similarity = 'euclidean'
        self.multimodal_merging = centroids_merging_strategy

        self.penalty_weight = penalty_weight
        self.sit_memory_size = sit_memory_size

        self.sit = sit

        if sit:

            if memory_type == 'random':
                self.storage_policy = RandomMemory(**memory_parameters)
            elif memory_type == 'clustering':
                self.storage_policy = ClusteringMemory(**memory_parameters)
            else:
                assert False, 'Unknown memory type '

            if merging_strategy == 'scale_translate':
                self.scaler = ScaleTranslate()
            elif merging_strategy == 'none':
                self.scaler = FakeMerging()
            else:
                self.scaler = Projector(merging_strategy)

            if centroids_merging_strategy is not None:
                if merging_strategy == 'scale_translate':
                    self.centroids_scaler = ScaleTranslate()
                elif merging_strategy == 'none':

                    self.centroids_scaler = FakeMerging()
                else:
                    self.centroids_scaler = Projector(merging_strategy)
            else:
                self.centroids_scaler = None

    def before_training_exp(self, strategy: "BaseStrategy",
                            num_workers: int = 0, shuffle: bool = True,
                            **kwargs):

        if len(self.tasks_centroids) == 0 or not self.sit:
            return

        strategy.dataloader = ReplayDataLoader(
            strategy.adapted_dataset,
            self.storage_policy.buffer,
            oversample_small_tasks=True,
            num_workers=num_workers,
            batch_size=strategy.train_mb_size,
            shuffle=shuffle)

    def calculate_centroids(self, strategy: BaseStrategy, dataset, model=None,
                            task=None):
        device = strategy.device

        if model is None:
            model = strategy.model

        if task is None:
            task = strategy.experience.current_experience

        if self.current_centroids is not None:
            centroids = torch.zeros_like(self.current_centroids)
        else:
            x = next(iter(
                    DataLoader(dataset, batch_size=strategy.train_mb_size)))[0]
            x = x.to(device)

            classes = strategy.experience.classes_in_this_experience

            centroids = avalanche_forward(model, x, task)[0].unsqueeze(0)
            centroids = centroids.expand(len(classes), -1)

            centroids = torch.zeros_like(centroids)

        dataloader = DataLoader(dataset, batch_size=strategy.train_mb_size)

        counter = torch.zeros(len(centroids), 1, device=device)

        for d in dataloader:
            x, y, tid = d
            x = x.to(device)
            y = y.to(device)

            embeddings = avalanche_forward(model, x, task)
            centroids = torch.index_add(centroids, 0, y, embeddings)
            counter = torch.index_add(counter, 0, y, torch.ones_like(y, dtype=counter.dtype)[:, None])

        centroids = centroids / counter

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
        loss_val = -log_p_y.gather(1, y.unsqueeze(-1))

        return loss_val

    def loss(self, strategy, **kwargs):
        if not strategy.model.training:
            return -1

        tid = strategy.experience.current_experience
        mb_output, y, x = strategy.mb_output, strategy.mb_y, strategy.mb_x
        tasks = strategy.mb_task_id

        centroids = self.calculate_centroids(strategy,
                                             strategy.experience.dev_dataset)
        self.current_centroids = centroids

        unique_tasks = torch.unique(tasks)
        if len(unique_tasks) == 1:

            loss_val = self._loss_f(mb_output, y, centroids).view(-1).mean()

        else:
            loss_val = 0

            if not self.sit or (self.sit and len(self.tasks_centroids) == 0):
                for task in unique_tasks:
                    task_mask = tasks == task
                    o_task = mb_output[task_mask]
                    y_task = y[task_mask]

                    if task < len(self.tasks_centroids):
                        centroids = self.tasks_centroids[task]

                    loss = self._loss_f(o_task, y_task, centroids).view(
                        -1).sum()
                    loss_val += loss
                loss_val = loss_val / len(x)

            else:

                offsets = np.cumsum([len(c) for c in self.tasks_centroids])

                offsets_tensor = torch.tensor([0] + offsets.tolist(),
                                              dtype=torch.long,
                                              device=strategy.device)

                y += torch.index_select(offsets_tensor, 0, tasks)

                embs = [avalanche_forward(strategy.model, x, task)
                        for task in range(len(self.tasks_centroids))] + \
                       [avalanche_forward(strategy.model, x, tid)]
                e = self.combine_embeddings(embs)

                centroids = self.tasks_centroids + [self.current_centroids]
                centroids = self.combine_centroids(centroids)

                loss_val = self._loss_f(e, y, centroids).view(-1).mean()

        return loss_val

    def combine_embeddings(self, embeddings):

        all_embs = [self.scaler(e, task) for task, e in enumerate(embeddings)]

        embeddings = torch.stack(all_embs, -1)

        embeddings = embeddings.mean(-1)

        return embeddings

    def combine_centroids(self, centroids):

        scaler = self.scaler if self.centroids_scaler is None else self.centroids_scaler

        centroids = [scaler(c, task) for task, c in enumerate(centroids)]
        centroids = torch.cat(centroids, 0)

        return centroids

    def after_training_exp(self, strategy, **kwargs):

        with torch.no_grad():
            tid = strategy.experience.current_experience

            self.tasks_centroids.append(self.current_centroids.detach())

            if isinstance(strategy.model, BatchNormModelWrap):
                for name, module in strategy.model.named_modules():
                    if isinstance(module, TaskIncrementalBatchNorm2D):
                        module.freeze_eval(tid)

        if self.sit:
            self.storage_policy.update(dataset=strategy.experience.dataset,
                                       tid=tid,
                                       model=strategy.model)

    def before_train_dataset_adaptation(self, strategy: 'BaseStrategy',
                                        **kwargs):

        self.past_model = deepcopy(strategy.model)
        self.past_model.eval()

        num_tasks = len(self.tasks_centroids)

        if num_tasks == 0 or not self.sit:
            return

        emb_shape = self.tasks_centroids[0].shape[1]

        if num_tasks == 1:
            self.scaler.add_task(embedding_size=emb_shape)

            if self.centroids_scaler is not None:
                self.centroids_scaler.add_task(embedding_size=emb_shape)

        self.scaler.add_task(embedding_size=emb_shape)
        self.scaler = self.scaler.to(strategy.device)

        if self.centroids_scaler is not None:
            self.centroids_scaler.add_task(embedding_size=emb_shape)

            self.centroids_scaler = self.centroids_scaler.to(strategy.device)

            strategy.optimizer.param_groups[0]['params'] = list(
                chain(strategy.model.parameters(),
                      self.scaler.parameters(),
                      self.centroids_scaler.parameters()))
        else:
            strategy.optimizer.param_groups[0]['params'] = list(
                chain(strategy.model.parameters(),
                      self.scaler.parameters()))

    @torch.no_grad()
    def calculate_classes(self, strategy, embeddings):
        strategy.model.eval()

        correct_task = strategy.experience.current_experience
        x = strategy.mb_x

        if self.sit and len(self.tasks_centroids) > 1:
            cumsum = np.cumsum([len(c) for c in self.tasks_centroids])

            upper = cumsum[correct_task]
            lower = 0 if correct_task == 0 else cumsum[correct_task - 1]

            embs = [avalanche_forward(strategy.model, x, task)
                    for task in range(len(self.tasks_centroids))]
            e = self.combine_embeddings(embs)

            centroids = self.combine_centroids(self.tasks_centroids)

            pred = self.calculate_similarity(e, centroids).argmax(-1)
            pred[pred >= upper] = -1
            pred = pred - lower

        else:
            if len(self.tasks_centroids) == 0:
                centroids = self.current_centroids
            else:
                centroids = self.tasks_centroids[correct_task]

            if centroids is None:
                return torch.full_like(strategy.mb_y, -1)

            sim = self.calculate_similarity(embeddings, centroids)
            pred = torch.argmax(sim, 1)

        return pred

    def before_backward(self, strategy, **kwargs):
        if len(self.tasks_centroids) > 0:

            x, y, t = strategy.mbatch
            dist = 0

            if self.penalty_weight > 0:
                for i in range(len(self.tasks_centroids)):
                    p_e = avalanche_forward(self.past_model, x, i)
                    c_e = avalanche_forward(strategy.model, x, i)

                    _dist = torch.norm(p_e - c_e, p=2, dim=1)

                    dist += _dist.mean()

                dist = dist / len(self.tasks_centroids)
                dist = dist * self.penalty_weight

            strategy.loss += dist

    @torch.no_grad()
    def save_embeddings(self, strategy, exps, path):
        strategy.model.eval()
        os.makedirs(path, exist_ok=True)

        for experience in exps:
            embs = []
            labels = []

            tid = experience.current_experience + 1

            for x, y, t in DataLoader(experience.dataset.eval(),
                                      batch_size=strategy.train_mb_size):
                labels.extend(y.tolist())

                x, y, t = x.to(strategy.device), \
                          y.to(strategy.device), \
                          t.to(strategy.device)

                embeddings = strategy.model(x, t)
                embeddings = embeddings.cpu().numpy()

                embs.append(embeddings)

            embs = np.concatenate(embs)
            n_tasks_so_far = len(self.tasks_centroids)
            p = os.path.join(path,
                             f'embeddings_current_task{n_tasks_so_far}_evaluated_task{tid}')

            np.save(p, embs)

            p = os.path.join(path,
                             f'labels_current_task{n_tasks_so_far}_evaluated_task{tid}')

            np.save(p, labels)

            centroids = [t.cpu().numpy() for t in self.tasks_centroids]
            centroids = np.stack(centroids, 0)

            p = os.path.join(path,
                             f'centroids_current_task{n_tasks_so_far}')

            np.save(p, centroids)

            if self.sit and len(self.tasks_centroids) > 1:
                lens = [len(c) for c in self.tasks_centroids]
                offsets = np.cumsum(lens)
                offsets_tensor = torch.tensor([0] + offsets.tolist(),
                                              dtype=torch.long,
                                              device=strategy.device)

                embs = []
                labels = []

                for x, y, t in DataLoader(experience.dataset.eval(),
                                          batch_size=strategy.train_mb_size):

                    x, y, t = x.to(strategy.device), \
                              y.to(strategy.device), \
                              t.to(strategy.device)

                    embs = [avalanche_forward(strategy.model, x, task)
                            for task in range(len(self.tasks_centroids))]
                    embeddings = self.combine_embeddings(embs)

                    embs.append(embeddings.cpu().numpy())

                    y += torch.index_select(offsets_tensor, 0, t)

                    labels.extend(y.cpu().tolist())

                embs = np.concatenate(embs)

                centroids = self.combine_centroids(self.tasks_centroids)

                centroids = centroids.cpu().numpy()

                p = os.path.join(path,
                                 f'ci_embeddings_current_task{n_tasks_so_far}_evaluated_task{tid}')

                np.save(p, embs)

                p = os.path.join(path,
                                 f'ci_labels_current_task{n_tasks_so_far}_evaluated_task{tid}')

                np.save(p, labels)

                p = os.path.join(path,
                                 f'ci_centroids_current_task{n_tasks_so_far}')

                np.save(p, centroids)
