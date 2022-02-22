from typing import Optional, List

import numpy as np
from avalanche.training import BaseStrategy
from avalanche.training.plugins import StrategyPlugin, EvaluationPlugin
from avalanche.training.plugins.evaluation import default_logger
from torch.optim import Optimizer
from torch.utils.data import Subset

from methods.plugins.cml import ContinualMetricLearningPlugin
from methods.plugins.er import EmbeddingRegularizationPlugin
from models.utils import MultiHeadBackbone, EmbeddingModelDecorator


class EmbeddingRegularization(BaseStrategy):

    def __init__(self, model: MultiHeadBackbone,
                 optimizer: Optimizer, criterion,
                 mem_size: int,
                 penalty_weight: float,
                 train_mb_size: int = 1,
                 train_epochs: int = 1,
                 eval_mb_size: int = None,
                 device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger,
                 eval_every=-1):

        rp = EmbeddingRegularizationPlugin(mem_size, penalty_weight)
        if plugins is None:
            plugins = [rp]
        else:
            plugins.append(rp)

        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device,
            plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)


class ContinualMetricLearning(BaseStrategy):

    def __init__(self, model: MultiHeadBackbone,
                 optimizer: Optimizer, criterion,
                 mem_size: int,
                 penalty_weight: float,
                 train_mb_size: int = 1,
                 train_epochs: int = 1,
                 eval_mb_size: int = None,
                 device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger,
                 eval_every=-1):

        rp = ContinualMetricLearningPlugin(mem_size, penalty_weight)
        if plugins is None:
            plugins = [rp]
        else:
            plugins.append(rp)

        self.dev_dataset = None

        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device,
            plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)

    def train_dataset_adaptation(self, **kwargs):
        """ Initialize `self.adapted_dataset`. """
        self.adapted_dataset = self.experience.dataset
        # self.adapted_dataset = self.adapted_dataset.train()
