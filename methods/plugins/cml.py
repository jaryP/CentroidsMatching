from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
from avalanche.benchmarks.utils import AvalancheSubset
from avalanche.training import BaseStrategy
from avalanche.training.plugins import StrategyPlugin
from torch import cosine_similarity, log_softmax, softmax
from torch.nn.functional import normalize
from torch.utils.data import DataLoader


class ContinualMetricLearningPlugin(StrategyPlugin):
    def __init__(self, penalty_weight: float):
        super().__init__()

        self.past_model = None
        self.penalty_weight = penalty_weight
        self.similarity = 'euclidean'
        self.tasks_centroids = {}

    def calculate_centroids(self, strategy: BaseStrategy, dataset):

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
            embeddings = strategy.model.forward_single_task(x, tid, False)
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
            centroids = self.last_centroids
            return 0
        else:
            centroids = self.calculate_centroids(strategy,
                                                 strategy.experience.dev_dataset)

        mb_output, y = strategy.mb_output, strategy.mb_y

        sim = self.calculate_similarity(mb_output, centroids)

        log_p_y = log_softmax(sim, dim=1)
        loss_val = -log_p_y.gather(1, y.unsqueeze(-1))
        loss_val = loss_val.view(-1).mean()

        return loss_val

    def after_training_exp(self, strategy, **kwargs):
        with torch.no_grad():
            tid = strategy.experience.current_experience

            centroids = self.calculate_centroids(strategy,
                                                 strategy.experience.dev_dataset)
            self.last_centroids = centroids

            self.tasks_centroids[tid] = centroids

            self.past_model = deepcopy(strategy.model)

            for param in strategy.model.classifier.classifiers[str(tid)].parameters():
                param.requires_grad_(False)

    def calculate_classes(self, strategy, embeddings):
        # centroids = self.calculate_centroids(strategy,
        #                                      strategy.experience.dev_dataset)
        centroids = self.tasks_centroids[strategy.experience.current_experience]
        sim = self.calculate_similarity(embeddings,
                                        centroids)
        sm = softmax(sim, -1)
        pred = torch.argmax(sm, 1)
        return pred

    # @torch.no_grad()
    # def update_memory(self, strategy, dataset, t, batch_size):
    #     dataloader = DataLoader(dataset.eval(), batch_size=batch_size)
    #     tot = 0
    #     device = strategy.device
    #
    #     for mbatch in dataloader:
    #         x, y, tid = mbatch[0].to(device), mbatch[1], mbatch[-1].to(device)
    #         emb, _ = strategy.model.forward_single_task(x, t, True)
    #
    #         emb = emb.detach().clone()
    #         x = x.detach().clone()
    #         tid = tid.detach().clone()
    #
    #         if tot + x.size(0) <= self.patterns_per_experience:
    #             if t not in self.memory_x:
    #                 self.memory_x[t] = x
    #                 self.memory_y[t] = emb
    #                 self.memory_tid[t] = tid.clone()
    #             else:
    #                 self.memory_x[t] = torch.cat((self.memory_x[t], x), dim=0)
    #                 self.memory_y[t] = torch.cat((self.memory_y[t], emb), dim=0)
    #                 self.memory_tid[t] = torch.cat((self.memory_tid[t], tid),
    #                                                dim=0)
    #
    #         else:
    #             diff = self.patterns_per_experience - tot
    #             if t not in self.memory_x:
    #                 self.memory_x[t] = x[:diff]
    #                 self.memory_y[t] = emb[:diff]
    #                 self.memory_tid[t] = tid[:diff]
    #             else:
    #                 self.memory_x[t] = torch.cat((self.memory_x[t], x[:diff]),
    #                                              dim=0)
    #                 self.memory_y[t] = torch.cat((self.memory_y[t], emb[:diff]),
    #                                              dim=0)
    #                 self.memory_tid[t] = torch.cat((self.memory_tid[t],
    #                                                 tid[:diff]), dim=0)
    #             break
    #         tot += x.size(0)

    def before_backward(self, strategy, **kwargs):
        if strategy.clock.train_exp_counter > 0:
            x, _, _ = strategy.mbatch
            dist = 0

            for i in range(len(self.tasks_centroids)):
                p_e = self.past_model.forward_single_task(x, i)
                c_e = strategy.model.forward_single_task(x, i)

                p_e_n = normalize(p_e)
                c_e_n = normalize(c_e)

                # p_e_n = normalize(past_centroids)
                # c_e_n = normalize(current_centroids)

                _dist = torch.norm(p_e_n - c_e_n, p=2, dim=1)
                dist += _dist.mean()

            dist = dist * self.penalty_weight

            strategy.loss += dist

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

    # def before_train_dataset_adaptation(self, strategy, **kwargs):
    #     dataset = strategy.experience.dataset
    #
    #     idx = np.arange(len(dataset))
    #     np.random.shuffle(idx)
    #     dev_i = int(len(idx) * 0.1)
    #
    #     dev_idx = idx[:dev_i]
    #     train_idx = idx[dev_i:]
    #
    #     dev = AvalancheSubset(dataset._original_dataset.eval(), dev_idx)
    #     train = AvalancheSubset(dataset._original_dataset, train_idx)
    #
    #     strategy.experience.dataset = train
    #     strategy.experience.dev_dataset = dev
