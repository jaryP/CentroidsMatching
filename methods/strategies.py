from collections import defaultdict
from typing import Optional, List, Sequence

import numpy as np
import torch
from avalanche.benchmarks import Experience
from avalanche.benchmarks.utils import AvalancheSubset, AvalancheConcatDataset
from avalanche.benchmarks.utils.data_loader import TaskBalancedDataLoader
from avalanche.models import DynamicModule
from avalanche.training import BaseStrategy
from avalanche.training.plugins import StrategyPlugin, EvaluationPlugin
from avalanche.training.plugins.evaluation import default_logger
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from methods.plugins.cml import CentroidsMatching, MemoryCentroidsMatching
from methods.plugins.cml_utils import BatchNormModelWrap
from methods.plugins.cope import ContinualPrototypeEvolution
from methods.plugins.er import EmbeddingRegularizationPlugin
from methods.plugins.ewc import EWCCustomPlugin
from methods.plugins.ssil import SeparatedSoftmax
from models.utils import CombinedModel


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

    def __init__(self,
                 feature_extractor: nn.Module,
                 classifier: nn.Module,
                 model: CombinedModel,
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

        for name, module in feature_extractor.named_modules():
            if isinstance(module, _BatchNorm):
                feature_extractor = BatchNormModelWrap(feature_extractor)
                break

        model = CombinedModel(feature_extractor, classifier)

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

    def __init__(self,
                 model: CombinedModel,
                 dev_split_size: float,
                 optimizer: Optimizer,
                 criterion, penalty_weight: float,
                 train_mb_size: int = 1,
                 train_epochs: int = 1, proj_w=1,
                 eval_mb_size: int = None,
                 device=None,
                 memory_parameters=None,
                 sit: bool = False, num_experiences: int = 20,
                 sit_memory_size: int = 500,
                 merging_strategy: str = 'scale_translate',
                 memory_type: str = 'random',
                 centroids_merging_strategy: str = None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger, eval_every=-1):

        if not sit and any(
                isinstance(module, _BatchNorm) for module in model.modules()):
            model = BatchNormModelWrap(model)

        rp = CentroidsMatching(penalty_weight, sit,
                               proj_w=proj_w,
                               memory_type=memory_type,
                               memory_parameters=memory_parameters,
                               merging_strategy=merging_strategy,
                               sit_memory_size=sit_memory_size,
                               centroids_merging_strategy=centroids_merging_strategy)

        self.rp = rp

        if plugins is None:
            plugins = [rp]
        else:
            plugins.append(rp)

        self.dev_split_size = dev_split_size
        self.dev_dataloader = None

        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device,
            plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)

    def train_dataset_adaptation(self, **kwargs):
        """ Initialize `self.adapted_dataset`. """

        exp_n = self.experience.current_experience

        if not hasattr(self.experience, 'dev_dataset'):
            dataset = self.experience.dataset

            idx = np.arange(len(dataset))
            np.random.shuffle(idx)

            if isinstance(self.dev_split_size, int):
                dev_i = self.dev_split_size
            else:
                dev_i = int(len(idx) * self.dev_split_size)

            dev_idx = idx[:dev_i]
            train_idx = idx[dev_i:]

            self.experience.dataset = AvalancheSubset(dataset.train(),
                                                      train_idx)
            self.experience.dev_dataset = AvalancheSubset(dataset.eval(),
                                                          dev_idx)

        self.adapted_dataset = self.experience.dataset

    def make_train_dataloader(self, num_workers=0, shuffle=True,
                              pin_memory=True, **kwargs):

        self.dataloader = DataLoader(
            self.adapted_dataset,
            num_workers=num_workers,
            batch_size=self.eval_mb_size,
            pin_memory=pin_memory)

    def criterion(self):
        """ Loss function. """
        loss = self.rp.loss(self)
        return loss

    def forward(self):
        res = super().forward()
        if not self.model.training:
            res = self.rp.calculate_classes(self, res)
        return res


class MemoryContinualMetricLearning(BaseStrategy):
    def __init__(self,
                 model: CombinedModel,
                 dev_split_size: float,
                 optimizer: Optimizer,
                 criterion,
                 train_mb_size: int = 1,
                 train_epochs: int = 1,
                 eval_mb_size: int = None,
                 device=None,
                 sit: bool = False,
                 memory_size: int = 500,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger, eval_every=-1):

        rp = MemoryCentroidsMatching(
            sit=sit,
            memory_size=memory_size)

        self.rp = rp

        if plugins is None:
            plugins = [rp]
        else:
            plugins.append(rp)

        self.dev_split_size = dev_split_size
        self.dev_dataloader = None

        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device,
            plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)

    def train_dataset_adaptation(self, **kwargs):
        """ Initialize `self.adapted_dataset`. """

        exp_n = self.experience.current_experience

        if not hasattr(self.experience, 'dev_dataset'):
            dataset = self.experience.dataset

            idx = np.arange(len(dataset))
            np.random.shuffle(idx)

            if isinstance(self.dev_split_size, int):
                dev_i = self.dev_split_size
            else:
                dev_i = int(len(idx) * self.dev_split_size)

            dev_idx = idx[:dev_i]
            train_idx = idx[dev_i:]

            self.experience.dataset = AvalancheSubset(dataset.train(),
                                                      train_idx)
            self.experience.dev_dataset = AvalancheSubset(dataset.eval(),
                                                          dev_idx)

        self.adapted_dataset = self.experience.dataset

    def make_train_dataloader(self, num_workers=0, shuffle=True,
                              pin_memory=True, **kwargs):

        self.dataloader = DataLoader(
            self.adapted_dataset,
            num_workers=num_workers,
            batch_size=self.eval_mb_size,
            pin_memory=pin_memory)

    def criterion(self):
        """ Loss function. """
        loss = self.rp.loss(self)
        return loss

    def forward(self):
        res = super().forward()
        if not self.model.training:
            res = self.rp.calculate_classes(self, res)
        return res


class SeparatedSoftmaxIncrementalLearning(BaseStrategy):

    def __init__(self,
                 model: CombinedModel,
                 optimizer: Optimizer,
                 criterion,
                 train_mb_size: int = 1,
                 train_epochs: int = 1,
                 eval_mb_size: int = None,
                 device=None,
                 sit_memory_size: int = 500,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger, eval_every=-1):

        rp = SeparatedSoftmax(mem_size=sit_memory_size)

        self.rp = rp

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


class CoPE(BaseStrategy):

    def __init__(self,
                 model: CombinedModel,
                 optimizer: Optimizer,
                 criterion,
                 train_mb_size: int = 1,
                 train_epochs: int = 1,
                 eval_mb_size: int = None,
                 device=None,
                 memory_size: int = 500,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger, eval_every=-1):

        self.experiences = None

        rp = ContinualPrototypeEvolution(memory_size=memory_size)

        self.rp = rp

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

    # def train(self, experiences,
    #           eval_streams=None,
    #           **kwargs):
    #     """ Training loop. if experiences is a single element trains on it.
    #     If it is a sequence, trains the model on each experience in order.
    #     This is different from joint training on the entire stream.
    #     It returns a dictionary with last recorded value for each metric.
    #
    #     :param experiences: single Experience or sequence.
    #     :param eval_streams: list of streams for evaluation.
    #         If None: use training experiences for evaluation.
    #         Use [] if you do not want to evaluate during training.
    #
    #     :return: dictionary containing last recorded value for
    #         each metric name.
    #     """
    #     self.is_training = True
    #     self._stop_training = False
    #
    #     self.model.train()
    #     self.model.to(self.device)
    #
    #     # Normalize training and eval data.
    #     if not isinstance(experiences, Sequence):
    #         experiences = [experiences]
    #
    #     if eval_streams is None:
    #         eval_streams = [experiences]
    #
    #     self._before_training(**kwargs)
    #
    #     self._periodic_eval(eval_streams, do_final=False, do_initial=True)
    #
    #     self.experiences = experiences
    #
    #     self.model.train()
    #
    #     if eval_streams is None:
    #         eval_streams = self.experiences
    #
    #     for i, exp in enumerate(eval_streams):
    #         if not isinstance(exp, Sequence):
    #             eval_streams[i] = [exp]
    #
    #     # Data Adaptation (e.g. add new samples/data augmentation)
    #     self._before_train_dataset_adaptation(**kwargs)
    #     self.train_dataset_adaptation(**kwargs)
    #     self._after_train_dataset_adaptation(**kwargs)
    #     self.make_train_dataloader(**kwargs)
    #
    #     self.experience = self.adapted_dataset
    #
    #     # Model Adaptation (e.g. freeze/add new units)
    #     self.model = self.model_adaptation()
    #     self.make_optimizer()
    #
    #     self._before_training_exp(**kwargs)
    #
    #     do_final = True
    #     if self.eval_every > 0 and \
    #             (self.train_epochs - 1) % self.eval_every == 0:
    #         do_final = False
    #
    #     self._before_training_epoch(**kwargs)
    #     self.training_epoch(**kwargs)
    #     self._after_training_epoch(**kwargs)
    #     self._periodic_eval(eval_streams, do_final=False)
    #
    #     # Final evaluation
    #     self._periodic_eval(eval_streams, do_final=do_final)
    #     self._after_training_exp(**kwargs)
    #
    #     self._after_training(**kwargs)
    #
    #     res = self.evaluator.get_last_metrics()
    #     return res
    #
    # def train_exp(self, experience: Experience, eval_streams=None, **kwargs):
    #     """ Training loop over a single Experience object.
    #
    #     :param experience: CL experience information.
    #     :param eval_streams: list of streams for evaluation.
    #         If None: use the training experience for evaluation.
    #         Use [] if you do not want to evaluate during training.
    #     :param kwargs: custom arguments.
    #     """
    #     self.experience = experience
    #     self.model.train()
    #
    #     if eval_streams is None:
    #         eval_streams = [experience]
    #     for i, exp in enumerate(eval_streams):
    #         if not isinstance(exp, Sequence):
    #             eval_streams[i] = [exp]
    #
    #     # Data Adaptation (e.g. add new samples/data augmentation)
    #     self._before_train_dataset_adaptation(**kwargs)
    #     self.train_dataset_adaptation(**kwargs)
    #     self._after_train_dataset_adaptation(**kwargs)
    #     self.make_train_dataloader(**kwargs)
    #
    #     # Model Adaptation (e.g. freeze/add new units)
    #     self.model = self.model_adaptation()
    #     self.make_optimizer()
    #
    #     self._before_training_exp(**kwargs)
    #
    #     do_final = True
    #     if self.eval_every > 0 and \
    #             (self.train_epochs - 1) % self.eval_every == 0:
    #         do_final = False
    #
    #     for _ in range(self.train_epochs):
    #         self._before_training_epoch(**kwargs)
    #
    #         if self._stop_training:  # Early stopping
    #             self._stop_training = False
    #             break
    #
    #         self.training_epoch(**kwargs)
    #         self._after_training_epoch(**kwargs)
    #         self._periodic_eval(eval_streams, do_final=False)
    #
    #     # Final evaluation
    #     self._periodic_eval(eval_streams, do_final=do_final)
    #     self._after_training_exp(**kwargs)
    #
    # def model_adaptation(self, model=None):
    #     """Adapts the model to the current data.
    #
    #     Calls the :class:`~avalanche.models.DynamicModule`s adaptation.
    #     """
    #     if model is None:
    #         model = self.model
    #
    #     for module in model.modules():
    #         if isinstance(module, DynamicModule):
    #             module.adaptation(self.experience)
    #     return model.to(self.device)
    #
    # def train_dataset_adaptation(self, **kwargs):
    #     """ Initialize `self.adapted_dataset`. """
    #     self.adapted_dataset = AvalancheConcatDataset([exp.dataset.train() for exp in self.experiences])

    def criterion(self):
        """ Loss function. """
        loss = self.rp.loss(self)
        return loss

    def forward(self):
        res = super().forward()
        res = torch.nn.functional.normalize(res, p=2, dim=1)
        if not self.model.training:
            # else:
            res = self.rp.calculate_classes(self, res)

        return res


class CustomEWC(BaseStrategy):
    """ Elastic Weight Consolidation (EWC) strategy.

    See EWC plugin for details.
    This strategy does not use task identities.
    """

    def __init__(self, model, optimizer: Optimizer, criterion,
                 ewc_lambda: float, mode: str = 'separate',
                 decay_factor: Optional[float] = None,
                 keep_importance_data: bool = False,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger, eval_every=-1):
        """ Init.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param ewc_lambda: hyperparameter to weigh the penalty inside the total
               loss. The larger the lambda, the larger the regularization.
        :param mode: `separate` to keep a separate penalty for each previous
               experience. `onlinesum` to keep a single penalty summed over all
               previous tasks. `onlineweightedsum` to keep a single penalty
               summed with a decay factor over all previous tasks.
        :param decay_factor: used only if mode is `onlineweightedsum`.
               It specify the decay term of the importance matrix.
        :param keep_importance_data: if True, keep in memory both parameter
                values and importances for all previous task, for all modes.
                If False, keep only last parameter values and importances.
                If mode is `separate`, the value of `keep_importance_data` is
                set to be True.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        """
        ewc = EWCCustomPlugin(ewc_lambda, mode, decay_factor,
                              keep_importance_data)
        if plugins is None:
            plugins = [ewc]
        else:
            plugins.append(ewc)

        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)
