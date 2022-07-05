from collections import defaultdict
from copy import deepcopy
from itertools import chain

import numpy as np
import torch
from avalanche.benchmarks.utils import AvalancheConcatDataset, AvalancheSubset
from avalanche.benchmarks.utils.data_loader import GroupBalancedDataLoader
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
    RandomMemory


class _CentroidsMatching(StrategyPlugin):
    def __init__(self, penalty_weight: float, sit=False,
                 proj_w: float = 1,
                 sit_memory_size: int = 200,
                 memory_type='random',
                 memory_parameters=None,
                 merging_strategy='scale_translate',
                 two_stage_sit=False):

        super().__init__()

        self.offsets_tensor = None
        self.proj_w = proj_w
        self.penalty_weight = penalty_weight
        self.sit_memory_size = sit_memory_size
        self.two_stage_sit = two_stage_sit

        self.samples_per_class = sit_memory_size

        if merging_strategy == 'scale_translate':
            self.scaler = ScaleTranslate()
            self.c_scaler = ScaleTranslate()
        elif merging_strategy == 'none':
            class FakeMerging(torch.nn.Module):
                def __init__(self, **kwargs):
                    super().__init__()

                def add_task(self, **kwargs):
                    pass

                def forward(self, x, i, **kwargs):
                    return x

            self.scaler = FakeMerging()
            self.c_scaler = FakeMerging()
        else:
            self.scaler = Projector(merging_strategy)

        self.c_scaler = Projector('mlp')

        self.past_model = None
        self.concatenated_dateset = None

        self.similarity = 'euclidean'
        self.tasks_centroids = []

        self.sit = sit

        self.current_centroids = None

        if memory_parameters is None:
            memory_parameters = {}

        if memory_type == 'random':
            self.memory = RandomMemory(**memory_parameters)
        elif memory_type == 'clustering':
            self.memory = ClusteringMemory(**memory_parameters)
        elif memory_type == 'distance_coverage':
            self.memory = DistanceMemory(**memory_parameters)
        elif memory_type == 'none':
            assert False, 'Unknown memory type '

        self.ws = None

    def custom_forward(self, model, x, task_id, t=None, force_eval=False):
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

        device = strategy.device
        dataloader = DataLoader(dataset,
                                batch_size=strategy.train_mb_size)

        classes = set(dataset.targets)

        embs = defaultdict(list)

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

        mb_output, y, x = strategy.mb_output, strategy.mb_y, strategy.mb_x
        t = strategy.mb_task_id
        tid = strategy.experience.current_experience

        centroids = self.calculate_centroids(strategy,
                                             strategy.experience.dev_dataset)

        self.current_centroids = centroids

        if False and self.sit and len(self.tasks_centroids) > 0:
            y += torch.index_select(self.offsets_tensor, 0, t)

            embs = [self.custom_forward(strategy.model, x, task)
                    for task in range(len(self.tasks_centroids))] + \
                   [self.custom_forward(strategy.model, x, tid)]

            e = self.combine_embeddings(embs)

            centroids = self.tasks_centroids + [self.current_centroids]
            centroids = self.combine_centroids(centroids)

            loss = self._loss_f(e, y, centroids)
            loss_val = loss.view(-1).mean()

        else:

            e = self.custom_forward(strategy.model, x,
                                    strategy.experience.current_experience)

            loss_val = self._loss_f(e, y, centroids).view(-1).mean()

        return loss_val

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

                    embeddings = 0

                    for task in range(len(self.tasks_centroids)):
                        embeddings += self.scaler(
                            self.custom_forward(strategy.model, x, task), task)

                    embs.append(embeddings.cpu().numpy())

                    y += torch.index_select(offsets_tensor, 0, t)

                    labels.extend(y.cpu().tolist())

                embs = np.concatenate(embs)

                centroids = torch.cat([self.scaler(c, task)
                                       for task, c in
                                       enumerate(self.tasks_centroids)], 0)

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

    def after_training_exp(self, strategy, **kwargs):

        with torch.no_grad():
            tid = strategy.experience.current_experience

            # centroids = self.calculate_centroids(strategy,
            #                                      strategy.experience.
            #                                      dev_dataset)

            # if self.sit:
            #     centroids = self.tasks_centroids + [self.current_centroids]
            #     centroids = self.combine_centroids(centroids)
            # else:

            centroids = self.current_centroids.detach()

            self.tasks_centroids.append(centroids)

            if isinstance(strategy.model, BatchNormModelWrap):
                for name, module in strategy.model.named_modules():
                    if isinstance(module, TaskIncrementalBatchNorm2D):
                        module.freeze_eval(tid)

            self.past_model = deepcopy(strategy.model)
            self.past_model.eval()

            lens = [len(c) for c in self.tasks_centroids]
            offsets = np.cumsum(lens)

            offsets_tensor = torch.tensor([0] + offsets.tolist(),
                                          dtype=torch.long,
                                          device=strategy.device)
            self.offsets_tensor = offsets_tensor

        if self.sit:
            self.past_model.train()
            # self.concatenated_dateset = self.memory.concatenated_dataset()
            datasets = [strategy.experience.dataset] \
                       + self.memory.buffer_datasets
            # datasets = [m for m in self.memory.memory.values()]

            concatenated_dateset = GroupBalancedDataLoader(datasets,
                                                           oversample_small_groups=True,
                                                           batch_size=strategy.train_mb_size,
                                                           shuffle=True)

            self.memory.add_task(strategy.experience.dataset,
                                 tid=tid,
                                 model=self.past_model)

            # datasets = [strategy.experience.dataset] + [m for m in
            #                                             self.memory.memory.values()]

            # datasets = AvalancheConcatDataset(
            #     [m for m in self.memory.memory.values()])
            #
            # concatenated_dateset = DataLoader(datasets,
            #                                   batch_size=10,
            #                                   shuffle=True)

            # self.concatenated_dateset = self.memory.concatenated_dataset()

            past_scaler = deepcopy(self.scaler)

            if len(self.tasks_centroids) > 1 and self.two_stage_sit:

                self.scaler.reset()

                for _ in range(len(self.tasks_centroids)):
                    self.scaler.add_task(self.tasks_centroids[0].shape[1])

                self.scaler = self.scaler.to(strategy.device)

                opt = Adam(chain(self.scaler.parameters(),
                                 self.ws.parameters(),
                                 strategy.model.parameters()),
                           lr=1e-3)

                # opt = deepcopy(strategy.optimizer)
                # opt.state = defaultdict(dict)
                #
                # opt.param_groups[0]['params'] = list(
                #     chain(self.scaler.parameters(),
                #           self.ws.parameters(),
                #           strategy.model.parameters()))

                best_score = 0
                best_model = (deepcopy(strategy.model.state_dict()),
                              deepcopy(self.scaler.state_dict()))

                for epoch in range(50):
                    losses = []
                    dev_tot, dev_corrects = 0, 0
                    train_tot, train_corrects = 0, 0

                    strategy.model.train()

                    # for x, y, t in self._get_reg_dataloader(strategy):
                    for x, y, t in concatenated_dateset:

                        x, y, t = x.to(strategy.device), y.to(
                            strategy.device), t.to(strategy.device)

                        y += torch.index_select(self.offsets_tensor, 0, t)

                        embs = [self.custom_forward(strategy.model, x, task)
                                for task in range(len(self.tasks_centroids))]
                        e = self.combine_embeddings(embs, True)

                        centroids = self.combine_centroids(self.tasks_centroids)

                        sim = self.calculate_similarity(e, centroids)
                        pred = torch.argmax(sim, -1)

                        train_tot += len(pred)
                        train_corrects += (pred == y).sum().item()

                        loss = self._loss_f(e, y, centroids)
                        loss = loss.view(-1).mean()

                        dist = 0

                        for i in range(len(self.tasks_centroids)):
                            p_e = self.custom_forward(self.past_model, x, i,
                                                      force_eval=True)

                            c_e = self.custom_forward(strategy.model, x, i,
                                                      force_eval=True)

                            # if i < len(self.tasks_centroids) - 1:
                            #     c_e = self.scaler(c_e, i)
                            #     p_e = past_scaler(p_e, i)

                            _dist = torch.norm(p_e - c_e, p=2, dim=1)
                            dist += _dist.mean()

                        dist = dist / (len(self.tasks_centroids) - 1)
                        dist = dist * self.penalty_weight
                        loss = loss + dist

                        losses.append(loss.item())

                        opt.zero_grad()
                        loss.backward()
                        opt.step()

                    strategy.model.eval()
                    with torch.no_grad():
                        for x, y, t in DataLoader(
                                strategy.experience.dev_dataset,
                                batch_size=strategy.train_mb_size):
                            x, y, t = x.to(strategy.device), y.to(
                                strategy.device), t.to(strategy.device)

                            y += torch.index_select(self.offsets_tensor, 0, t)

                            embs = [self.custom_forward(strategy.model, x, task)
                                    for task in
                                    range(len(self.tasks_centroids))]
                            e = self.combine_embeddings(embs)

                            centroids = self.combine_centroids(
                                self.tasks_centroids)

                            sim = self.calculate_similarity(e, centroids)
                            pred = torch.argmax(sim, -1)

                            dev_tot += len(pred)
                            dev_corrects += (pred == y).sum().item()

                    print(epoch,
                          np.mean(losses),
                          dev_tot,
                          dev_corrects,
                          dev_corrects / dev_tot,
                          train_tot,
                          train_corrects,
                          train_corrects / train_tot)

                    if best_score < train_corrects / train_tot:
                        print('Best model',
                              best_score,
                              dev_corrects / dev_tot,
                              train_corrects / train_tot)

                        best_score = train_corrects / train_tot
                        best_model = (deepcopy(strategy.model.state_dict()),
                                      deepcopy(self.scaler.state_dict()))

                strategy.model.load_state_dict(best_model[0])
                self.scaler.load_state_dict(best_model[1])

            self.past_model.eval()

    def before_train_dataset_adaptation(self, strategy: 'BaseStrategy',
                                        **kwargs):

        num_tasks = len(self.tasks_centroids)

        if num_tasks == 0 or not self.sit:
            return

        emb_shape = self.tasks_centroids[0].shape[1]

        # self.scaler.reset()
        # self.centroids_scaler.reset()
        #
        # for i in range(num_tasks + 1):
        #     self.scaler.add_task(emb_shape)
        #     self.centroids_scaler.add_task(emb_shape)

        if num_tasks == 1:
            self.scaler.add_task(embedding_size=emb_shape)
            self.c_scaler.add_task(emb_shape)

        self.scaler.add_task(embedding_size=emb_shape)
        self.c_scaler.add_task(emb_shape)

        self.ws = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(emb_shape * (num_tasks + 1),
                            (emb_shape * (num_tasks + 1)) // 2),
            torch.nn.ReLU(),
            torch.nn.Linear((emb_shape * (num_tasks + 1)) // 2, emb_shape),
        ).to(strategy.device)

        # self.ws = torch.nn.Linear(emb_shape * (num_tasks + 1), num_tasks + 1) \
        #     .to(strategy.device)

        # self.centroids_scaler.add_task(emb_shape)

        self.scaler = self.scaler.to(strategy.device)
        self.c_scaler = self.c_scaler.to(strategy.device)

        strategy.optimizer.state = defaultdict(dict)

        strategy.optimizer.param_groups[0]['params'] = list(
            chain(strategy.model.parameters(),
                  # self.scaler.parameters(),
                  # self.ws.parameters(),
                  # self.c_scaler.parameters()
                  ))

        self.concatenated_dateset = iter(self._get_reg_dataloader(strategy))

    def combine_embeddings(self, embeddings, train=False):

        all_embs = [self.scaler(torch.dropout(embeddings[t], 0.5, train), t)
                    for t in range(len(embeddings))]

        # all_embs = [self.scaler(e, task) for task, e in enumerate(embeddings)]

        # all_embs += [embeddings[-1]]
        # all_embs = torch.stack(all_embs, -1)

        # embeddings = torch.cat(all_embs, -1)
        # embeddings = self.ws(embeddings)

        embeddings = torch.stack(all_embs, -1).sum(-1)

        # embeddings = torch.cat(embeddings, -1)

        # ws = torch.softmax(ws, -1)

        # all_embs = ws[:, None] * all_embs

        # embeddings = all_embs.sum(-1)

        return embeddings

    def combine_centroids(self, centroids):
        centroids = [self.scaler(c, task) for task, c in enumerate(centroids)]

        # centroids = [self.scaler(centroids[t], t)
        #              for t in range(len(centroids) - 1)] + [centroids[-1]]

        centroids = torch.cat(centroids, 0)

        return centroids

    @torch.no_grad()
    def calculate_classes(self, strategy, embeddings):
        correct_task = strategy.experience.current_experience
        x = strategy.mb_x

        if correct_task > len(self.tasks_centroids):
            return torch.full_like(strategy.mb_y, -1)

        if self.sit and len(self.tasks_centroids) > 1:
            cumsum = np.cumsum([len(c) for c in self.tasks_centroids])

            upper = cumsum[correct_task]
            lower = 0 if correct_task == 0 else cumsum[correct_task - 1]

            embs = [self.custom_forward(strategy.model, x, task)
                    for task in range(len(self.tasks_centroids))]
            e = self.combine_embeddings(embs)

            centroids = self.combine_centroids(self.tasks_centroids)

            pred = self.calculate_similarity(e, centroids).argmax(-1)

            # we scale the prediction into the trange of the task,
            # in order to simplify the evaluation process

            pred[pred >= upper] = -1
            pred = pred - lower

        else:
            if len(self.tasks_centroids) == correct_task:
                centroids = self.current_centroids
            else:
                centroids = self.tasks_centroids[correct_task]

            if centroids is None:
                return torch.full_like(strategy.mb_y, -1)

            sim = self.calculate_similarity(embeddings, centroids)
            pred = torch.argmax(sim, 1)

        return pred

    # def before_training_exp(self, strategy: "BaseStrategy",
    #                         num_workers: int = 0, shuffle: bool = True,
    #                         **kwargs):
    #     """
    #     Dataloader to build batches containing examples from both memories and
    #     the training dataset
    #     """
    #     if len(self.tasks_centroids) == 0:
    #         # first experience. We don't use the buffer, no need to change
    #         # the dataloader.
    #         return
    #
    #     strategy.dataloader = ReplayDataLoader(
    #         strategy.adapted_dataset,
    #         self.memory.concatenated_dataset(),
    #         oversample_small_tasks=True,
    #         num_workers=num_workers,
    #         batch_size=strategy.train_mb_size,
    #         shuffle=shuffle)

    def _get_reg_dataloader(self, strategy, batch_size=None):

        batch_size = batch_size if batch_size is not None else strategy.train_mb_size
        return DataLoader(AvalancheConcatDataset(self.memory.buffer_datasets),
                          batch_size=batch_size,
                          shuffle=True)

        datasets = [strategy.experience.dataset] + [m for m in
                                                    self.memory.memory.values()]
        # datasets = [m for m in self.memory.memory.values()]

        concatenated_dateset = GroupBalancedDataLoader(datasets,
                                                       oversample_small_groups=True,
                                                       batch_size=batch_size,
                                                       shuffle=True)

        return concatenated_dateset

    def before_backward(self, strategy, **kwargs):

        if len(self.tasks_centroids) > 0:

            x, y, t = strategy.mbatch
            dist = 0
            tid = strategy.experience.current_experience

            # current_epoch = strategy.clock.train_exp_epochs
            # decay = 1 - np.exp(
            #     -current_epoch / (max(strategy.train_epochs // 2, 1)))
            decay = 1

            # decay = (2 ** (len(strategy.dataloader) + 1 -
            #                strategy.clock.train_epoch_iterations))
            # decay /= (2 ** len(strategy.dataloader) - 1)
            # decay = 1 - decay

            if self.sit \
                    and self.proj_w > 0 \
                    and decay > 0 \
                    and not self.two_stage_sit:

                # loader = DataLoader(self.memory.concatenated_dataset(),
                #                     batch_size=len(x),
                #                     shuffle=True)

                try:
                    past_x, past_y, past_t = next(self.concatenated_dateset)
                except StopIteration:
                    # if ret is None:
                    self.concatenated_dateset = iter(
                        self._get_reg_dataloader(strategy))
                    past_x, past_y, past_t = next(self.concatenated_dateset)

                past_x, past_y, past_t = past_x.to(strategy.device), past_y.to(
                    strategy.device), past_t.to(strategy.device)

                x, y, t = past_x, past_y, past_t

                # if len(x) > len(past_x):
                #     x = x[:len(past_x)]
                #     y = y[:len(past_x)]
                #     t = t[:len(past_x)]
                #
                # if len(past_x) > len(x):
                #     past_x = past_x[:len(x)]
                #     past_y = past_y[:len(x)]
                #     past_t = past_t[:len(x)]

                x, y, t = torch.cat((x, past_x)), \
                          torch.cat((y, past_y)), \
                          torch.cat((t, past_t))

                # p = torch.full((y.shape[0],), 0.5, device=x.device)
                # m = torch.bernoulli(p)
                #
                # xm = m.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                # x = xm * x + (1 - xm) * past_x
                #
                # m = m.long()
                #
                # y = m * y + (1 - m) * past_y
                # t = m * t + (1 - m) * past_t

                y += torch.index_select(self.offsets_tensor, 0, t)

                embs = [self.custom_forward(strategy.model, x, task)
                        for task in range(len(self.tasks_centroids))] + \
                       [self.custom_forward(strategy.model, x, tid)]

                e = self.combine_embeddings(embs)

                centroids = self.tasks_centroids + [self.current_centroids]
                centroids = self.combine_centroids(centroids)

                loss = self._loss_f(e, y, centroids)
                loss = loss.view(-1).mean()
                loss = loss * self.proj_w * decay

                print(loss, strategy.loss)

                strategy.loss += loss

            if self.penalty_weight > 0:
                for i in range(len(self.tasks_centroids)):
                    p_e = self.custom_forward(self.past_model, x, i,
                                              force_eval=True)

                    c_e = self.custom_forward(strategy.model, x, i,
                                              force_eval=True)

                    _dist = torch.norm(p_e - c_e, p=2, dim=1)
                    dist += _dist.mean()

                dist = dist / len(self.tasks_centroids)
                dist = dist * self.penalty_weight

            strategy.loss += dist


class __CentroidsMatching(StrategyPlugin):
    def __init__(self, penalty_weight: float, sit=False,
                 proj_w: float = 1,
                 sit_memory_size: int = 200,
                 memory_type='random',
                 memory_parameters=None,
                 merging_strategy='scale_translate',
                 two_stage_sit=False):

        super().__init__()

        self.proj_w = proj_w
        self.penalty_weight = penalty_weight
        self.sit_memory_size = sit_memory_size
        self.two_stage_sit = two_stage_sit

        self.samples_per_class = sit_memory_size

        if merging_strategy == 'scale_translate':
            self.scaler = ScaleTranslate()
        else:
            self.scaler = Projector(merging_strategy)

        self.past_model = None
        self.concatenated_dateset = None

        self.similarity = 'euclidean'
        self.tasks_centroids = []

        self.sit = sit

        self.current_centroids = None

        if memory_parameters is None:
            memory_parameters = {}

        if memory_type == 'random':
            self.memory = RandomMemory(**memory_parameters)
        elif memory_type == 'clustering':
            self.memory = ClusteringMemory(**memory_parameters)
        elif memory_type == 'distance_coverage':
            self.memory = DistanceMemory(**memory_parameters)
        else:
            assert False, 'Unknown memory type '

    def custom_forward(self, model, x, task_id, t=None, force_eval=False):
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

        device = strategy.device
        dataloader = DataLoader(dataset,
                                batch_size=strategy.train_mb_size)

        classes = set(dataset.targets)

        embs = defaultdict(list)

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

        mb_output, y, x = strategy.mb_output, strategy.mb_y, strategy.mb_x
        t = strategy.mb_task_id

        centroids = self.calculate_centroids(strategy,
                                             strategy.experience.dev_dataset)

        self.current_centroids = centroids

        if self.sit and len(self.tasks_centroids) > 0:

            lens = [len(c) for c in self.tasks_centroids]
            offsets = np.cumsum(lens)

            offsets_tensor = torch.tensor([0] + offsets.tolist(),
                                          dtype=torch.long,
                                          device=strategy.device)

            y += torch.index_select(offsets_tensor, 0, t)
            e = 0

            for task in range(len(offsets_tensor)):
                e += self.scaler(self.custom_forward(strategy.model, x, task),
                                 task)

            centroids = torch.cat([self.scaler(c, task)
                                   for task, c in
                                   enumerate(chain(self.tasks_centroids,
                                                   [self.current_centroids]))],
                                  0)

            loss = self._loss_f(e, y, centroids)
            loss_val = loss.view(-1).mean()

        else:

            mb_output = self.custom_forward(strategy.model, x,
                                            strategy.experience.current_experience)

            loss_val = self._loss_f(mb_output, y, centroids).view(-1).mean()

        return loss_val

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

                    embeddings = 0

                    for task in range(len(self.tasks_centroids)):
                        embeddings += self.scaler(
                            self.custom_forward(strategy.model, x, task), task)

                    embs.append(embeddings.cpu().numpy())

                    y += torch.index_select(offsets_tensor, 0, t)

                    labels.extend(y.cpu().tolist())

                embs = np.concatenate(embs)

                centroids = torch.cat([self.scaler(c, task)
                                       for task, c in
                                       enumerate(self.tasks_centroids)], 0)

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

    def after_training_exp(self, strategy, **kwargs):

        tid = strategy.experience.current_experience

        centroids = self.calculate_centroids(strategy,
                                             strategy.experience.
                                             dev_dataset).detach()

        self.tasks_centroids.append(centroids)

        if isinstance(strategy.model, BatchNormModelWrap):
            for name, module in strategy.model.named_modules():
                if isinstance(module, TaskIncrementalBatchNorm2D):
                    module.freeze_eval(tid)

        self.past_model = deepcopy(strategy.model)
        self.past_model.eval()

        if self.sit:
            self.memory.add_task(strategy.experience.eval_dataset, tid=tid,
                                 model=self.past_model)

            self.concatenated_dateset = self.memory.concatenated_dataset()

            if len(self.tasks_centroids) > 1 and self.two_stage_sit:
                dataset = AvalancheConcatDataset(
                    [self.memory.concatenated_dataset()])

                opt = Adam(self.scaler.parameters())

                lens = [len(c) for c in self.tasks_centroids]
                offsets = np.cumsum(lens)

                offsets_tensor = torch.tensor([0] + offsets.tolist(),
                                              dtype=torch.long,
                                              device=strategy.device)
                for i in range(10):
                    losses = []
                    for x, y, t in DataLoader(dataset, batch_size=10):
                        x, y, t = x.to(strategy.device), y.to(
                            strategy.device), t.to(strategy.device)

                        y += torch.index_select(offsets_tensor, 0, t)
                        e = 0

                        for task in range(len(self.tasks_centroids)):
                            e += self.scaler(
                                self.custom_forward(strategy.model, x, task),
                                task)

                        centroids = torch.cat([self.scaler(c, task)
                                               for task, c in
                                               enumerate(self.tasks_centroids)],
                                              0)

                        loss = self._loss_f(e, y, centroids)
                        loss = loss.view(-1).mean()
                        losses.append(loss.item())

                        opt.zero_grad()
                        loss.backward()
                        opt.step()

                    print(np.mean(losses))

    def before_train_dataset_adaptation(self, strategy: 'BaseStrategy',
                                        **kwargs):

        num_tasks = len(self.tasks_centroids)

        if num_tasks == 0 or not self.sit:
            return

        emb_shape = self.tasks_centroids[0].shape[1]

        if num_tasks == 1:
            self.scaler.add_task(emb_shape)

        self.scaler.add_task(emb_shape)

        self.scaler = self.scaler.to(strategy.device)

        strategy.optimizer.state = defaultdict(dict)

        strategy.optimizer.param_groups[0]['params'] = list(
            chain(strategy.model.parameters(),
                  self.scaler.parameters()))

        # self.concatenated_dateset = TaskBalancedDataLoader(
        #     AvalancheConcatDataset([self.memory.concatenated_dataset(),
        #                             strategy.experience.dataset]),
        #     oversample_small_tasks=True, batch_size=strategy.train_mb_size // (
        #                 len(self.tasks_centroids) + 1))

        # self.concatenated_dateset = GroupBalancedDataLoader(
        #     chain(strategy.experience.dataset, self.memory.memory.values()),
        #     oversample_small_groups=True,
        #     batch_size=strategy.train_mb_size,
        #     shuffle=True)

        self.concatenated_dateset = iter(self._get_reg_dataloader(strategy))

        # self.concatenated_dateset = TaskBalancedDataLoader(self.memory.concatenated_dataset(), shuffle=True, batch_size=strategy.train_mb_size)

    # def after_train_dataset_adaptation(self, strategy, **kwargs):
    #     if len(self.memory.concatenated_dataset()) > 0:
    #         strategy.adapted_dataset = AvalancheConcatDataset([
    #             self.memory.concatenated_dataset(),
    #             strategy.experience.dataset.train()])

    @torch.no_grad()
    def calculate_classes(self, strategy, embeddings):
        mode = strategy.model.training
        strategy.model.eval()

        correct_task = strategy.experience.current_experience
        x = strategy.mb_x

        if correct_task > len(self.tasks_centroids):
            return torch.full_like(strategy.mb_y, -1)

        if self.sit and len(self.tasks_centroids) > 1:
            cumsum = np.cumsum([len(c) for c in self.tasks_centroids])

            upper = cumsum[correct_task]
            lower = 0 if correct_task == 0 else cumsum[correct_task - 1]

            e = 0

            for task in range(len(self.tasks_centroids)):
                e += self.scaler(
                    self.custom_forward(strategy.model, x, task),
                    task)

            centroids = torch.cat([self.scaler(c, task)
                                   for task, c in
                                   enumerate(self.tasks_centroids)],
                                  0)

            pred = self.calculate_similarity(e, centroids).argmax(-1)

            # we scale the prediction into the trange of the task,
            # in order to simplify the evaluation process

            pred[pred >= upper] = -1
            pred = pred - lower

        else:
            if len(self.tasks_centroids) == correct_task:
                centroids = self.current_centroids
            else:
                centroids = self.tasks_centroids[correct_task]

            if centroids is None:
                return torch.full_like(strategy.mb_y, -1)

            sim = self.calculate_similarity(embeddings, centroids)
            pred = torch.argmax(sim, 1)

        strategy.model.train(mode)

        return pred

    def _get_reg_dataloader(self, strategy):
        datasets = [strategy.experience.dataset] + [m for m in
                                                    self.memory.memory.values()]

        concatenated_dateset = GroupBalancedDataLoader(datasets,
                                                       oversample_small_groups=True,
                                                       batch_size=strategy.train_mb_size // len(
                                                           datasets),
                                                       shuffle=True)

        return concatenated_dateset

    def before_training_exp(self, strategy: "BaseStrategy",
                            num_workers: int = 0, shuffle: bool = True,
                            **kwargs):
        if len(self.memory.memory) == 0:
            return
        datasets = [strategy.adapted_dataset] + [m for m in
                                                 self.memory.memory.values()]
        strategy.dataloader = GroupBalancedDataLoader(datasets,
                                                      oversample_small_groups=True,
                                                      batch_size=strategy.train_mb_size // len(
                                                          datasets),
                                                      shuffle=shuffle,
                                                      num_workers=num_workers)

    def before_backward(self, strategy, **kwargs):
        if len(self.tasks_centroids) > 0:

            x, y, t = strategy.mbatch

            if self.penalty_weight > 0:
                dist = 0

                for i in range(len(self.tasks_centroids)):
                    p_e = self.custom_forward(self.past_model, x, i,
                                              force_eval=True)

                    c_e = self.custom_forward(strategy.model, x, i,
                                              force_eval=True)

                    _dist = torch.norm(p_e - c_e, p=2, dim=1)
                    dist += _dist.mean()

                dist = dist / len(self.tasks_centroids)

                dist = dist * self.penalty_weight

                strategy.loss += dist


class CentroidsMatching(StrategyPlugin):
    def __init__(self,
                 penalty_weight: float,
                 sit=False,
                 proj_w: float = 1,
                 sit_memory_size: int = 200,
                 memory_type='random',
                 memory_parameters=None,
                 merging_strategy='scale_translate',
                 **kwargs):

        super().__init__()

        self.proj_w = proj_w
        self.scaler = ScaleTranslate()
        self.embs_proj = None

        self.forests = {}
        self.classifier = None
        self.past_model = None

        self.penalty_weight = penalty_weight
        self.sit_memory_size = sit_memory_size

        self.similarity = 'euclidean'

        self.tasks_centroids = []

        self.sit = sit

        self.current_centroids = None

        if memory_type == 'random':
            self.memory = RandomMemory(**memory_parameters)
        elif memory_type == 'clustering':
            self.memory = ClusteringMemory(**memory_parameters)
        elif memory_type == 'distance_coverage':
            self.memory = DistanceMemory(**memory_parameters)
        elif memory_type == 'none':
            assert False, 'Unknown memory type '

        self.storage_policy = ExperienceBalancedBuffer(
            max_size=500,
            adaptive_size=True)

    def custom_forward(self, model, x, task_id, t=None, force_eval=False):
        f = model(x, task_id)
        if t is None:
            return f

        return f, [f]

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
            x = next(iter(DataLoader(dataset, batch_size=strategy.train_mb_size)))[0]
            x = x.to(device)
            classes = strategy.experience.classes_in_this_experience

            centroids = avalanche_forward(model, x, task)[0].unsqueeze(0)
            centroids = centroids.expand(len(classes), -1)

            centroids = torch.zeros_like(centroids)

        # if self.sit:
        #     strategy.model.classifier.classifiers[str(task)].force_eval()

        # model = strategy.model
        dataloader = DataLoader(dataset, batch_size=strategy.train_mb_size)
        # batch_size=len(dataset))

        classes = set(dataset.targets)

        embs = defaultdict(list)

        # classes = set()
        # for x, y, tid in data:
        #     x, y, _ = d
        #     emb, _ = strategy.model.forward_single_task(x, t, True)
        #     classes.update(y.detach().cpu().tolist())

        classes = sorted(classes)
        counter = torch.zeros(len(centroids), 1, device=device)

        for d in dataloader:
            x, y, tid = d
            x = x.to(device)
            y = y.to(device)

            embeddings = avalanche_forward(model, x, task)
            centroids = torch.index_add(centroids, 0, y, embeddings)
            counter = torch.index_add(counter, 0, y,
                                      torch.ones_like(y, dtype=counter.dtype))

            # for c in classes:
            #     embs[c].append(embeddings[y == c])

        centroids = centroids / counter

        # embs = {c: torch.cat(e, 0) for c, e in embs.items()}
        #
        # centroids = torch.stack([torch.mean(embs[c], 0) for c in classes], 0)

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
        loss_val = -log_p_y.gather(1, y.unsqueeze(-1))

        return loss_val

    def loss(self, strategy, **kwargs):
        if not strategy.model.training:
            return -1

        tid = strategy.experience.current_experience
        mb_output, y, x = strategy.mb_output, strategy.mb_y, strategy.mb_x
        centroids = self.calculate_centroids(strategy,
                                             strategy.experience.dev_dataset)

        # if self.sit and len(self.tasks_centroids) > 0:
        #     t = strategy.mb_task_id
        #
        #     lens = [len(c) for c in self.tasks_centroids]
        #     offsets = np.cumsum(lens)
        #
        #     offsets_tensor = torch.tensor([0] + offsets.tolist(),
        #                                   dtype=torch.long,
        #                                   device=strategy.device)
        #
        #     y += torch.index_select(offsets_tensor, 0, t)
        #
        #     e = self.embs_proj(
        #         [self.custom_forward(strategy.model, x, task)
        #          for task in range(len(offsets_tensor))])
        #
        #     centroids = torch.cat([self.scaler(c, task)
        #                            for task, c in
        #                            enumerate(chain(self.tasks_centroids,
        #                                            [centroids]))], 0)
        #
        #     loss = self._loss_f(e, y, centroids)
        #     loss_val = loss.view(-1).mean()
        #
        # else:
        # if self.sit and len(self.tasks_centroids) > 0:
        #     c = torch.cat(self.tasks_centroids, 0)
        #     centroids = torch.cat((c, centroids), 0)

        self.current_centroids = centroids

        mb_output = avalanche_forward(strategy.model, x, tid)

        if self.sit and len(self.tasks_centroids) > 0:
            mb_output = self.scaler(mb_output, tid)
            centroids = self.scaler(centroids, tid)

        loss_val = self._loss_f(mb_output, y, centroids).view(-1).mean()

        return loss_val

    def combine_embeddings(self, embeddings, train=False):

        # all_embs = [self.scaler(embeddings[t], t)
        #             for t in range(len(embeddings) - 1)] + [embeddings[-1]]

        all_embs = [self.scaler(e, task) for task, e in enumerate(embeddings)]

        # all_embs += [embeddings[-1]]
        # all_embs = torch.stack(all_embs, -1)

        # embeddings = torch.cat(all_embs, -1)
        # embeddings = self.ws(embeddings)

        embeddings = torch.stack(all_embs, -1).sum(-1)

        # embeddings = torch.cat(embeddings, -1)

        # ws = torch.softmax(ws, -1)

        # all_embs = ws[:, None] * all_embs

        # embeddings = all_embs.sum(-1)

        return embeddings

    def combine_centroids(self, centroids):
        # centroids = [self.scaler(centroids[t], t)
        #             for t in range(len(centroids) - 1)] + [centroids[-1]]

        centroids = [self.scaler(c, task) for task, c in enumerate(centroids)]

        # centroids = [self.scaler(centroids[t], t)
        #              for t in range(len(centroids) - 1)] + [centroids[-1]]

        centroids = torch.cat(centroids, 0)

        return centroids

    def after_training_exp(self, strategy, **kwargs):

        with torch.no_grad():
            tid = strategy.experience.current_experience

            self.tasks_centroids.append(self.current_centroids.detach())

            if isinstance(strategy.model, BatchNormModelWrap):
                for name, module in strategy.model.named_modules():
                    # if isinstance(module, (TaskIncrementalBatchNorm2D,
                    #                        ClassIncrementalBatchNorm2D)):
                    if isinstance(module, TaskIncrementalBatchNorm2D):
                        module.freeze_eval(tid)

        if self.sit:
            # self.storage_policy.update(strategy, **kwargs)
            self.memory.add_task(strategy.experience.dataset, tid=tid,
                                 model=self.past_model)

            # idxs = np.arange(len(strategy.experience.dataset))
            # # idxs = idxs[np.concatenate((s[:self.sit_memory_size // 2],
            # #                             s[-self.sit_memory_size // 2:]))]
            # np.random.shuffle(idxs)
            # idxs = idxs[:self.sit_memory_size]
            #
            # dataset = AvalancheSubset(strategy.experience.dataset, idxs)
            #
            # self.memory[len(self.memory)] = dataset
            #
            # num_tasks = len(self.tasks_centroids)
            # if num_tasks == 1:
            #     return

    def before_train_dataset_adaptation(self, strategy: 'BaseStrategy',
                                        **kwargs):

        self.past_model = deepcopy(strategy.model)
        self.past_model.eval()

        num_tasks = len(self.tasks_centroids)

        if num_tasks == 0 or not self.sit:
            return

        emb_shape = self.tasks_centroids[0].shape[1]

        self.scaler.reset()

        for _ in range(num_tasks + 1):
            self.scaler.add_task(emb_shape)

        # if num_tasks == 1:
        #     self.scaler.add_task(emb_shape)
        #
        # self.scaler.add_task(emb_shape)

        # for layer in strategy.model.children():
        #     if hasattr(layer, 'reset_parameters'):
        #         layer.reset_parameters()

        # for _ in range(num_tasks + 1):
        #     self.scaler.add_task(emb_shape)

        # self.scaler.reset(num_tasks + 1, emb_shape)
        self.scaler = self.scaler.to(strategy.device)

        # self.embs_proj = EmbsProjector(emb_shape, num_tasks + 1).to(
        #     strategy.device)

        strategy.optimizer.state = defaultdict(dict)

        # strategy.optimizer.param_groups[0]['params'] = list(
        #     chain(strategy.model.parameters(),
        #           self.scaler.parameters(), self.embs_proj.parameters()))

        strategy.optimizer.param_groups[0]['params'] = list(
            chain(strategy.model.parameters(),
                  self.scaler.parameters()))

        # strategy.dataloader = ReplayDataLoader(
        #     strategy.adapted_dataset,
        #     self.storage_policy.buffer,
        #     oversample_small_tasks=True,
        #     batch_size=strategy.train_mb_size,
        #     shuffle=True)

    # def before_training_exp(self, strategy: "BaseStrategy",
    #                         num_workers: int = 0, shuffle: bool = True,
    #                         **kwargs):
    #
    #     if len(self.storage_policy.buffer) == 0:
    #         # first experience. We don't use the buffer, no need to change
    #         # the dataloader.
    #         return
    #
    #     strategy.dataloader = ReplayDataLoader(
    #         strategy.adapted_dataset,
    #         self.storage_policy.buffer,
    #         oversample_small_tasks=True,
    #         batch_size=strategy.train_mb_size,
    #         shuffle=True)

    @torch.no_grad()
    def calculate_classes(self, strategy, embeddings):
        strategy.model.eval()

        correct_task = strategy.experience.current_experience
        x = strategy.mb_x

        if self.sit and len(self.tasks_centroids) > 1:
            cumsum = np.cumsum([len(c) for c in self.tasks_centroids])

            upper = cumsum[correct_task]
            lower = 0 if correct_task == 0 else cumsum[correct_task - 1]

            # emb_shape = self.tasks_centroids[0].shape[1]
            # num_tasks = len(self.tasks_centroids)

            embs = [avalanche_forward(strategy.model, x, task)
                    for task in range(len(self.tasks_centroids))]
            e = self.combine_embeddings(embs)

            centroids = self.combine_centroids(self.tasks_centroids)

            # e = 0
            #
            # for task in range(len(self.tasks_centroids)):
            #     e += self.scaler(
            #         self.custom_forward(strategy.model, x, task),
            #         task)
            #
            # # e += self.scaler(self.custom_forward(strategy.model, x, task),
            # #                  task)
            #
            # # e = self.embs_proj([self.custom_forward(strategy.model, x, task)
            # #                     for task in range(num_tasks)])
            #
            # # centroids = torch.cat([c + self.scaler[task]
            # #                        for task, c in
            # #                        enumerate(self.tasks_centroids)],
            # #                       0)
            #
            # centroids = torch.cat([self.scaler(c, task)
            #                        for task, c in
            #                        enumerate(self.tasks_centroids)],
            #                       0)

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
            tot_sim = 0
            tid = strategy.experience.current_experience

            if self.sit and self.proj_w > 0 and strategy.train_epochs / (
                    strategy.clock.train_exp_epochs + 1) > 0.5:

                lens = [len(c) for c in self.tasks_centroids]
                offsets = np.cumsum(lens)

                offsets_tensor = torch.tensor([0] + offsets.tolist(),
                                              dtype=torch.long,
                                              device=strategy.device)

                # concatenated_tasks = self.storage_policy.buffer
                concatenated_tasks = self.memory.buffer

                loader = DataLoader(concatenated_tasks,
                                    batch_size=len(x),
                                    shuffle=True)

                past_x, past_y, past_t = next(iter(loader))
                past_x, past_y, past_t = past_x.to(strategy.device), past_y.to(
                    strategy.device), past_t.to(strategy.device)
                # x, y, t = torch.cat((x, past_x), 0), torch.cat(
                #     (y, past_y)), torch.cat((t, past_t))

                p = torch.full((y.shape[0],), 0.5, device=x.device)
                m = torch.bernoulli(p)

                xm = m.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                new_x = xm * x + (1 - xm) * past_x

                m = m.long()
                new_y = m * y + (1 - m) * past_y

                new_t = m * t + (1 - m) * past_t

                new_y += torch.index_select(offsets_tensor, 0, new_t)

                embs = [avalanche_forward(strategy.model, new_x, task)
                        for task in range(len(self.tasks_centroids))] + \
                       [avalanche_forward(strategy.model, new_x, len(self.tasks_centroids))]

                e = self.combine_embeddings(embs)

                centroids = self.tasks_centroids + [self.current_centroids]
                centroids = self.combine_centroids(centroids)

                # for task in range(len(offsets_tensor)):
                #     e += self.scaler(
                #         self.custom_forward(strategy.model, x, task),
                #         task)
                #
                # # e = self.embs_proj(
                # #     [self.custom_forward(strategy.model, x, task)
                # #      for task in range(len(offsets_tensor))])
                #
                # centroids = torch.cat([self.scaler(c, task)
                #                        for task, c in
                #                        enumerate(chain(self.tasks_centroids,
                #                                        [
                #                                            self.current_centroids]))],
                #                       0)

                loss = self._loss_f(e, new_y, centroids)
                loss = loss.view(-1).mean()

                strategy.loss += loss * self.proj_w

            if self.penalty_weight > 0:
                dists = []

                mode = strategy.model.training
                strategy.model.eval()

                for i in range(len(self.tasks_centroids)):
                    p_e = avalanche_forward(self.past_model, x, i)
                    c_e = avalanche_forward(strategy.model, x, i)

                    _dist = torch.norm(p_e - c_e, p=2, dim=1).mean()
                    dists.append(_dist)
                    # dist += _dist.mean()

                strategy.model.train(mode)

                dist = torch.tensor(dists).mean()
                # dist = dist / len(self.tasks_centroids)

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

                    embeddings = 0

                    for task in range(len(self.tasks_centroids)):
                        embeddings += self.scaler(
                            avalanche_forward(strategy.model, x, task), task)

                    embs.append(embeddings.cpu().numpy())

                    y += torch.index_select(offsets_tensor, 0, t)

                    labels.extend(y.cpu().tolist())

                embs = np.concatenate(embs)

                centroids = torch.cat([self.scaler(c, task)
                                       for task, c in
                                       enumerate(self.tasks_centroids)], 0)

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
