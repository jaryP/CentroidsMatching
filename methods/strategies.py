from typing import Optional, List

import numpy as np
from avalanche.benchmarks.utils.data_loader import TaskBalancedDataLoader
from avalanche.training import BaseStrategy
from avalanche.training.plugins import StrategyPlugin, EvaluationPlugin
from avalanche.training.plugins.evaluation import default_logger
from torch.optim import Optimizer
from torch.utils.data import Subset, DataLoader

from methods.plugins.cml import ContinualMetricLearningPlugin
from methods.plugins.er import EmbeddingRegularizationPlugin
from models.utils import MultiHeadBackbone, EmbeddingModelDecorator, \
    CombinedModel


class CustomSubset:
    def __init__(self, dataset, indices) -> None:
        self._dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    @property
    def dataset(self):
        return self._dataset

    def __getattr__(self, item):
        if item == 'dataset':
            a = getattr(self, item)
        else:
            a = getattr(self.dataset, item)
        return a


class EmbeddingRegularization(BaseStrategy):

    def __init__(self, model: CombinedModel,
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

    def __init__(self, model: CombinedModel, dev_split_size: float,
                 optimizer: Optimizer, criterion, penalty_weight: float,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = None, device=None,
                 sit: bool = False,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger, eval_every=-1):

        rp = ContinualMetricLearningPlugin(penalty_weight, sit)
        self.rp = rp
        if plugins is None:
            plugins = [rp]
        else:
            plugins.append(rp)

        self.dev_split_size = dev_split_size
        self.dev_dataloader = None
        self.dev_indexes = dict()

        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device,
            plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)

    # def eval_dataset_adaptation(self, **kwargs):
    #     """ Initialize `self.adapted_dataset`. """
    #     self.train_dataset_adaptation()
        # self.adapted_dataset = self.experience.dataset
        # self.adapted_dataset = self.adapted_dataset.eval()

    def train_dataset_adaptation(self, **kwargs):
        """ Initialize `self.adapted_dataset`. """

        exp_n = self.experience.current_experience

        if not hasattr(self.experience, 'dev-dataset'):

            dataset = self.experience.dataset
            idx = np.arange(len(dataset))
            np.random.shuffle(idx)
            dev_i = int(len(idx) * self.dev_split_size)

            dev_idx = idx[:dev_i]
            train_idx = idx[dev_i:]
            self.dev_indexes[exp_n] = (train_idx, dev_idx)

            self.experience.dataset = CustomSubset(dataset.train(), train_idx)
            self.experience.dev_dataset = CustomSubset(dataset.eval(), dev_idx)

        # if exp_n not in self.dev_indexes:
        #     train = self.experience.dataset
        #     idx = np.arange(len(train))
        #     np.random.shuffle(idx)
        #     dev_i = int(len(idx) * self.dev_split_size)
        #
        #     dev_idx = idx[:dev_i]
        #     train_idx = idx[dev_i:]
        #     self.dev_indexes[exp_n] = (train_idx, dev_idx)
        #
        #     dataset = self.experience.dataset
        #     self.experience.dataset = CustomSubset(dataset.train(), train_idx)
        #     self.experience.dev_dataset = CustomSubset(dataset.eval(), dev_idx)
        # else:
        #     train_idx, dev_idx = self.dev_indexes[exp_n]

        self.adapted_dataset = self.experience.dataset
        # self.adapted_dataset = self.adapted_dataset.train()

    # def make_train_dataloader(self, num_workers=0, shuffle=True,
    #                           pin_memory=True, **kwargs):
    #
    #     exp_n = self.experience.current_experience
    #     if exp_n not in self.dev_indexes:
    #         train = self.experience.dataset
    #         idx = np.arange(len(train))
    #         np.random.shuffle(idx)
    #         dev_i = int(len(idx) * self.dev_split_size)
    #
    #         dev_idx = idx[:dev_i]
    #         train_idx = idx[dev_i:]
    #         self.dev_indexes[exp_n] = (train_idx, dev_idx)
    #     else:
    #         train_idx, dev_idx = self.dev_indexes[exp_n]
    #
    #     self.dataloader = DataLoader(Subset(self.adapted_dataset, train_idx),
    #                                  num_workers=num_workers,
    #                                  batch_size=self.train_mb_size,
    #                                  shuffle=shuffle,
    #                                  pin_memory=pin_memory)
    #
    #     self.dev_dataloader = DataLoader(Subset(self.adapted_dataset.eval(),
    #                                             dev_idx),
    #                                      num_workers=num_workers,
    #                                      batch_size=self.train_mb_size,
    #                                      shuffle=shuffle,
    #                                      pin_memory=pin_memory)

    def criterion(self):
        """ Loss function. """
        loss = self.rp.loss(self)
        return loss

    def forward(self):
        res = super().forward()
        if not self.model.training:
            res = self.rp.calculate_classes(self, res)
        return res

    def make_train_dataloader(self, num_workers=0, shuffle=True,
                              pin_memory=True, **kwargs):
        """ Data loader initialization.

        Called at the start of each learning experience after the dataset
        adaptation.

        :param num_workers: number of thread workers for the data loading.
        :param shuffle: True if the data should be shuffled, False otherwise.
        :param pin_memory: If True, the data loader will copy Tensors into CUDA
            pinned memory before returning them. Defaults to True.
        """
        self.dataloader = DataLoader(
            self.adapted_dataset,
            num_workers=num_workers,
            batch_size=self.train_mb_size,
            shuffle=shuffle,
            pin_memory=pin_memory)
