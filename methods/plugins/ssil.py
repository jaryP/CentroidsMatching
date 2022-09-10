from copy import deepcopy

import torch
from avalanche.training import ClassBalancedBuffer, BaseStrategy
from avalanche.training.plugins import StrategyPlugin
from torch import softmax, log_softmax
from torch.nn.functional import kl_div
from torch.utils.data import DataLoader


class SeparatedSoftmax(StrategyPlugin):
    def __init__(self,
                 mem_size: int):

        super().__init__()

        self.past_model = None
        self.seen_classes = []

        self.memory = ClassBalancedBuffer(max_size=mem_size)
        self.loss = torch.nn.CrossEntropyLoss(reduction='sum')

    def after_training_exp(self, strategy, **kwargs):
        self.memory.update(strategy)
        self.seen_classes.append(strategy.experience.classes_in_this_experience)

    def before_train_dataset_adaptation(self, strategy: 'BaseStrategy',
                                        **kwargs):

        self.past_model = deepcopy(strategy.model)
        self.past_model.eval()

    def before_backward(self, strategy: 'BaseStrategy', **kwargs):
        if strategy.clock.train_exp_counter > 0:
            classes = strategy.experience.classes_in_this_experience
            n_classes = len(classes)
            T = 2

            dataloader = DataLoader(self.memory.buffer,
                                    batch_size=strategy.train_mb_size,
                                    shuffle=True)

            target = strategy.mb_y
            x = strategy.mb_x
            t = strategy.mb_task_id

            bs = len(x)

            px, py, pt = next(iter(dataloader))
            px, py, pt = px.to(strategy.device), \
                         py.to(strategy.device), \
                         pt.to(strategy.device)

            x = torch.cat((x, px))
            t = torch.cat((t, pt))
            y = torch.cat((target, py))

            output = strategy.model(x, t)

            curr = output[:bs, -n_classes:]
            curr_ce = self.loss(curr, y[:bs] - min(classes))

            prev_ce = self.loss(output[bs:, :-n_classes], y[bs:])
            ce = (curr_ce + prev_ce) / len(x)

            score = self.past_model(x, t)
            kd = 0

            for t in range(len(self.seen_classes)):
                s = min(self.seen_classes[t])
                e = max(self.seen_classes[t]) + 1

                soft_target = softmax(score[:, s:e] / T, dim=1)
                output_log = log_softmax(output[:, s:e] / T,
                                         dim=1)

                kd += kl_div(output_log, soft_target,
                             reduction='batchmean') * (T ** 2)

            strategy.loss = kd + ce
