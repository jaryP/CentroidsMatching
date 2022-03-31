from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.models import MultiTaskModule
from avalanche.training import BaseStrategy
from avalanche.training.plugins import StrategyPlugin
from sklearn.neighbors import LocalOutlierFactor
from torch import cosine_similarity, log_softmax, softmax, nn
from torch.nn import BatchNorm2d
from torch.nn.functional import normalize
from torch.utils.data import DataLoader


def wrap_model(model: nn.Module):
    for name, module in model.named_children():
        if isinstance(module, BatchNorm2d):
            setattr(model, name, CLBatchNorm2D(module))
        else:
            wrap_model(module)


class CLBatchNorm2D(MultiTaskModule):
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
                          device=self.device,
                          track_running_stats=self.track_running_stats,
                          eps=self.eps)

        self.bns[str(cbn)] = nbn

    def set_task(self, t):
        self.current_task = t

    def freeze_eval(self, t):
        self.freeze_bn[str(t)] = True
        self.bns[str(t)].eval()

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


class WrappedModel(MultiTaskModule):
    def forward_single_task(self, x: torch.Tensor,
                            task_label: int, **kwargs) -> torch.Tensor:

        return self.model.forward_single_task(x, task_label)

    def __init__(self, model: nn.Module):
        super().__init__()

        self.model = model
        wrap_model(self.model)

    def forward(self, x, task_labels, **kwargs):
        if not isinstance(task_labels, int):
            task_labels = torch.unique(task_labels)
            if len(task_labels) == 1:
                task_labels = task_labels.item()
            else:
                assert False

        for module in self.model.modules():
            if isinstance(module, CLBatchNorm2D):
                module.set_task(task_labels)

        return self.model(x, task_labels)


class ContinualMetricLearningPlugin(StrategyPlugin):
    def __init__(self, penalty_weight: float, sit=False):
        super().__init__()

        self.past_model = None
        self.penalty_weight = penalty_weight
        self.similarity = 'euclidean'
        self.tasks_centroids = {}
        self.sit = sit
        self.tasks_forest = {}

    @staticmethod
    def calculate_centroids(strategy: BaseStrategy, dataset):

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
            embeddings = strategy.model(x, tid)
            for c in classes:
                embs[c].append(embeddings[y == c])

        embs = {c: torch.cat(e, 0) for c, e in embs.items()}

        centroids = torch.stack([torch.mean(embs[c], 0) for c in classes], 0)
        return centroids

    def calculate_similarity(self, x, y, similarity: str = None, sigma=1):
        # if isinstance(y, list):
        #     if isinstance(y[0], MultivariateNormal):
        #         diff = torch.stack([-likelihhod(n, x) for n in y], 1)
        #         # diff = torch.stack([-n.log_prob(x) ** 10 for n in y], 1)
        #         return diff
        #     elif isinstance(y[0], tuple):
        #         diff = torch.stack([-likelihhod(n, x) for n in y], 1)
        #         return diff

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
            dist = cosine_similarity(a, b, -1) ** 2
        else:
            assert False

        return dist

    def loss(self, strategy, **kwargs):
        if not strategy.model.training:
            # centroids = self.last_centroids¬\
            return 0
        else:
            centroids = self.calculate_centroids(strategy,
                                                 strategy.experience.dev_dataset)
        # if strategy.experience.current_experience > 0:
        #     print(torch.isnan(centroids).sum())

        mb_output, y = strategy.mb_output, strategy.mb_y

        sim = self.calculate_similarity(mb_output, centroids)

        log_p_y = log_softmax(sim, dim=1)
        loss_val = -log_p_y.gather(1, y.unsqueeze(-1))
        loss_val = loss_val.view(-1).mean()

        return loss_val

    @torch.no_grad()
    def after_training_exp(self, strategy, **kwargs):
        tid = strategy.experience.current_experience

        centroids = self.calculate_centroids(strategy,
                                             strategy.experience.
                                             dev_dataset)

        self.tasks_centroids[tid] = centroids

        self.past_model = deepcopy(strategy.model)
        self.past_model.eval()

        if isinstance(strategy.model, WrappedModel):
            for name, module in strategy.model.named_modules():
                if isinstance(module, CLBatchNorm2D):
                    module.freeze_eval(tid)

        # for param in strategy.model.classifier.classifiers[str(tid)].parameters():
        #     param.requires_grad_(False)

        if self.sit:
            dataloader = DataLoader(strategy.experience.dev_dataset,
                                    batch_size=strategy.train_mb_size)
            embeddings = []
            for x, _, _ in dataloader:
                x = x.to(strategy.device)

                # e = strategy.model.forward_single_task(x, tid, False)
                e = strategy.model(x, tid)

                embeddings.extend(e.cpu().tolist())

            embeddings = np.asarray(embeddings)

            f = LocalOutlierFactor(novelty=True, n_neighbors=50)
            f.fit(embeddings)

            self.tasks_forest[tid] = f

    @torch.no_grad()
    def calculate_classes(self, strategy, embeddings):

        correct_task = strategy.experience.current_experience
        if False and self.sit and len(self.tasks_centroids) > 1:

            correct_embs = None
            predictions = []
            od = []

            for ti, forest in self.tasks_forest.items():
                e = strategy.model.forward(strategy.mb_x, ti, False)

                if ti == correct_task:
                    correct_embs = e

                e = e.cpu().numpy()

                forest_prediction = forest.score_samples(e)
                od.append(forest.predict(e))
                predictions.append(forest_prediction)

            predicted_tasks = np.argmax(predictions, 0)

            correct_prediction_mask = predicted_tasks == correct_task
            correct_prediction_mask = torch.tensor(correct_prediction_mask,
                                                   device=strategy.device,
                                                   dtype=torch.long)

            centroids = self.tasks_centroids[correct_task]
            sim = self.calculate_similarity(correct_embs, centroids)
            sm = softmax(sim, -1)
            sm = torch.argmax(sm, 1)

            pred = correct_prediction_mask * sm + (
                    1 - correct_prediction_mask) * -1

        else:

            centroids = self.tasks_centroids[correct_task]
            sim = self.calculate_similarity(embeddings, centroids)
            sm = softmax(sim, -1)
            pred = torch.argmax(sm, 1)
        return pred

    def before_backward(self, strategy, **kwargs):
        if strategy.clock.train_exp_counter > 0:

            # strategy.model.eval()

            # a = strategy.model.model.backbone.model.bn1.bns['0'].weight
            # b = self.past_model.model.backbone.model.bn1.bns['0'].weight

            x, _, tdi = strategy.mbatch
            dist = 0

            for i in range(len(self.tasks_centroids)):
                p_e = self.past_model.forward(x, i)
                c_e = strategy.model.forward(x, i)

                if False:
                    p_e = normalize(p_e)
                    c_e = normalize(c_e)

                _dist = torch.norm(p_e - c_e, p=2, dim=1)
                dist += _dist.mean()

            dist = dist * self.penalty_weight

            strategy.loss += dist

            # strategy.model.train()

            # tdi = strategy.experience.current_experience
            # if self.sit:
            #     tot_sim = 0
            #     p_e = strategy.model.forward_single_task(x, tdi)
            #
            #     for i in range(tdi):
            #         # for j in range(i + 1, len(self.tasks_centroids)):
            #         #     p_e = strategy.model.forward_single_task(x, i)
            #         c_e = strategy.model.forward_single_task(x, i)
            #
            #         _dist = torch.norm(p_e - c_e, p=2, dim=1)
            #         sim = 1 / (1 + _dist)
            #
            #         tot_sim += sim.mean()
            #
            #     strategy.loss += tot_sim * self.penalty_weight

    #         total_dis = 0
    #         device = strategy.device
    #
    #         strategy.model.train()
    #         for t in range(strategy.clock.train_exp_counter):
    #             strategy.model.train()
    #
    #             xref = self.memory_x[t].to(device)
    #             past_emb = self.memory_y[t].to(device)
    #
    #             embedding, _ = strategy.model.forward_single_task(
    #                 xref,
    #                 self.memory_tid[t],
    #                 True)
    #
    #             sim = cosine_similarity(past_emb, embedding) ** 2
    #             sim = sim.mean()
    #
    #             dis = 1 - sim
    #             total_dis += dis
    #
    #         total_dis = total_dis / strategy.clock.train_exp_counter
    #         strategy.loss += total_dis * self.penalty_weight
