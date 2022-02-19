from collections import defaultdict

import numpy as np
import torch
from avalanche.benchmarks.utils import AvalancheSubset
from avalanche.training import BaseStrategy
from avalanche.training.plugins import StrategyPlugin
from torch.utils.data import DataLoader


class ContinualMetricLearningPlugin(StrategyPlugin):
    def __init__(self, mem_size: int, penalty_weight: float):
        super().__init__()

        self.patterns_per_experience = mem_size
        self.penalty_weight = penalty_weight
        self.memory_x, self.memory_y, self.memory_tid = {}, {}, {}

    def calculate_centroids(self,
                            strategy: BaseStrategy):

        # model = strategy.model
        device = strategy.device
        eval_dataset = strategy.experience.dev_dataset

        dataloader = DataLoader(eval_dataset,
                                batch_size=strategy.train_mb_size)
        classes = set(eval_dataset.targets)

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

        centroids = torch.stack([torch.mean(embs[c], 0) for c in classes],
                                0)
        return centroids

    def after_training_exp(self, strategy, **kwargs):
        self.calculate_centroids(strategy)
        # self.update_memory(strategy,
        #                    strategy.experience.dataset,
        #                    strategy.clock.train_exp_counter,
        #                    strategy.train_mb_size)

    @torch.no_grad()
    def update_memory(self, strategy, dataset, t, batch_size):
        dataloader = DataLoader(dataset.eval(), batch_size=batch_size)
        tot = 0
        device = strategy.device

        for mbatch in dataloader:
            x, y, tid = mbatch[0].to(device), mbatch[1], mbatch[-1].to(device)
            emb, _ = strategy.model.forward_single_task(x, t, True)

            emb = emb.detach().clone()
            x = x.detach().clone()
            tid = tid.detach().clone()

            if tot + x.size(0) <= self.patterns_per_experience:
                if t not in self.memory_x:
                    self.memory_x[t] = x
                    self.memory_y[t] = emb
                    self.memory_tid[t] = tid.clone()
                else:
                    self.memory_x[t] = torch.cat((self.memory_x[t], x), dim=0)
                    self.memory_y[t] = torch.cat((self.memory_y[t], emb), dim=0)
                    self.memory_tid[t] = torch.cat((self.memory_tid[t], tid),
                                                   dim=0)

            else:
                diff = self.patterns_per_experience - tot
                if t not in self.memory_x:
                    self.memory_x[t] = x[:diff]
                    self.memory_y[t] = emb[:diff]
                    self.memory_tid[t] = tid[:diff]
                else:
                    self.memory_x[t] = torch.cat((self.memory_x[t], x[:diff]),
                                                 dim=0)
                    self.memory_y[t] = torch.cat((self.memory_y[t], emb[:diff]),
                                                 dim=0)
                    self.memory_tid[t] = torch.cat((self.memory_tid[t],
                                                    tid[:diff]), dim=0)
                break
            tot += x.size(0)

    # def before_backward(self, strategy, **kwargs):
    #     if strategy.clock.train_exp_counter > 0:
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

    def before_train_dataset_adaptation(self, strategy, **kwargs):
        dataset = strategy.experience.dataset

        idx = np.arange(len(dataset))
        np.random.shuffle(idx)
        dev_i = int(len(idx) * 0.1)

        dev_idx = idx[:dev_i]
        train_idx = idx[dev_i:]

        dev = AvalancheSubset(dataset._original_dataset.eval(), dev_idx)
        train = AvalancheSubset(dataset._original_dataset, train_idx)

        strategy.experience.dataset = train
        strategy.experience.dev_dataset = dev
