from typing import Optional, List

import numpy as np
from avalanche.training import BaseStrategy
from avalanche.training.plugins import StrategyPlugin, EvaluationPlugin
from avalanche.training.plugins.evaluation import default_logger
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from methods.plugins.cml import ContinualMetricLearningPlugin, \
    BatchNormModelWrap, DropContinualMetricLearningPlugin, \
    ClassIncrementalBatchNormModelWrap
from methods.plugins.er import EmbeddingRegularizationPlugin
from methods.plugins.hal import AnchorLearningPlugin
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

    def __init__(self, model: CombinedModel, dev_split_size: float,
                 optimizer: Optimizer, criterion, penalty_weight: float,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = None, device=None,
                 sit: bool = False, num_experiences: int = 20,
                 sit_memory_size: int = 200,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger, eval_every=-1):

        # if sit:
        #     rp = ClassIncrementalContinualMetricLearningPlugin(penalty_weight, sit)
        # else:

        # if any(isinstance(module, _BatchNorm) for module in
        #        model.modules()) and not sit:
        #     # for name, module in model.named_modules():
        #     #     if isinstance(module, _BatchNorm):
        #     model = BatchNormModelWrap(model)
        #     # break

        if any(isinstance(module, _BatchNorm) for module in
               model.modules()):
            # for name, module in model.named_modules():
            #     if isinstance(module, _BatchNorm):
            if sit:
                pass
                # model = ClassIncrementalBatchNormModelWrap(model)
            else:
                model = BatchNormModelWrap(model)
            # break

        # rp = ContinualMetricLearningPlugin(penalty_weight, sit,
        #                                    num_experiences=num_experiences,
        #                                    sit_memory_size=sit_memory_size)

        rp = DropContinualMetricLearningPlugin(penalty_weight, sit,
                                               num_experiences=num_experiences,
                                               sit_memory_size=sit_memory_size)

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


class AnchorLearning(BaseStrategy):
    def __init__(self,
                 feature_extractor: nn.Module,
                 classifier: nn.Module,
                 optimizer: Optimizer, criterion,
                 ring_size: int,
                 lamb: float,
                 beta: float,
                 alpha: float,
                 embedding_strength: float,
                 k: int = 100,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger, eval_every=-1,
                 **kwargs):

        model = CombinedModel(feature_extractor,
                              classifier=classifier)

        rp = AnchorLearningPlugin(ring_size=ring_size,
                                  regularization=lamb,
                                  decay_rate=beta,
                                  lr=alpha,
                                  embedding_strength=embedding_strength,
                                  k=k)
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

    # def training_epoch(self, **kwargs):
    #     """ Training epoch.
    #
    #     :param kwargs:
    #     :return:
    #     """
    #     for self.mbatch in self.dataloader:
    #         if self._stop_training:
    #             break
    #
    #         self._unpack_minibatch()
    #         self._before_training_iteration(**kwargs)
    #
    #         self.optimizer.zero_grad()
    #         self.loss = 0
    #
    #         # Forward
    #         self._before_forward(**kwargs)
    #         self.mb_output = self.forward()
    #         self._after_forward(**kwargs)
    #
    #         # Loss & Backward
    #         self.loss += self.criterion()
    #
    #         self._before_backward(**kwargs)
    #         self.loss.backward(retain_graph=False)
    #         self._after_backward(**kwargs)
    #
    #         # Optimization step
    #         self._before_update(**kwargs)
    #         self.optimizer.step()
    #         self._after_update(**kwargs)
    #
    #         self._after_training_iteration(**kwargs)

# class BatchNormModelWrapper(MultiTaskModule):
#     def __init__(self, model: nn.Module):
#         super().__init__()
#
#         self.model = model
#         self.task_bn = nn.ModuleDict()
#         self.current_task = None
#
#         # current_bn = nn.ModuleDict()
#         #
#         # for name, module in model.named_modules():
#         #     if isinstance(module, _BatchNorm):
#         #         name = name.replace('.', '_')
#         #         current_bn[name] = module
#         #
#         # self.task_bn['0'] = current_bn
#
#     def adaptation(self, dataset: AvalancheDataset = None):
#         # def sequential():
#         #     for i, l in enumerate(s):
#         #         if isinstance(l, BatchNorm2d):
#         #             if skip_last and i == len(s) - 1:
#         #                 continue
#         #             s[i] = wrapper_fn(l)
#         #
#         #         elif isinstance(l, BasicBlock):
#         #             s[i] = ResNetBlockWrapper(l, wrapper_fn)
#
#         def get_children(model: torch.nn.Module):
#             # get children form model!
#             children = list(model.named_children())
#             flatt_children = []
#             if children == []:
#                 # if model has no children; model is last child! :O
#                 return model
#             else:
#                 # look for children from children... to the last child!
#                 for anme, child in children:
#                     try:
#                         flatt_children.extend(get_children(child))
#                     except TypeError:
#                         flatt_children.append(get_children(child))
#             return flatt_children
#
#         if self.training:
#             current_bn = nn.ModuleDict()
#             # a = get_children(self.model)
#
#             for name, module in dict(self.model.named_children()).items():
#                 if isinstance(module, BatchNorm2d):
#                     nbn = BatchNorm2d(module.num_features)
#                     setattr(self.model, name, nbn)
#
#                     name = name.replace('.', '_')
#                     current_bn[name] = nbn
#
#             self.task_bn[str(len(self.task_bn))] = current_bn
#
#         # for name, module in dict(self.model.named_modules()).items():
#         #     if isinstance(module, _BatchNorm):
#         #         name = name.replace('.', '_')
#         #         # self.task_bn[i] = deepcopy(module)
#         #         setattr(self.model, name, bns[name])
#
#     def forward_single_task(self, x: torch.Tensor, task_label: int,
#                             return_embeddings: bool = False):
#
#         if self.current_task is None or task_label != self.current_task:
#             bns = self.task_bn[str(task_label)]
#
#             self.current_task = task_label
#
#             for name, module in dict(self.model.named_modules()).items():
#                 if isinstance(module, _BatchNorm):
#                     name = name.replace('.', '_')
#                     # self.task_bn[i] = deepcopy(module)
#                     setattr(self.model, name, bns[name])
#
#         return self.model(x=x, task_labels=task_label)
#
#     def forward(self, x, task_labels, **kwargs):
#
#         if isinstance(task_labels, int):
#             # fast path. mini-batch is single task.
#             return self.forward_single_task(x, task_labels)
#         else:
#             unique_tasks = torch.unique(task_labels)
#             if len(unique_tasks) == 1:
#                 unique_tasks = unique_tasks.item()
#                 return self.forward_single_task(x, unique_tasks)
#
#         assert False
#         # bns = self.task_bn[str(task_labels)]
#         #
#         # if self.current_task is None or task_labels != self.current_task:
#         #     self.current_task = task_labels
#         # # else:
#         # #     if task_label != self.current_task:
#         # #         self.current_task = task_label
#         # #
#         #     for name, module in self.model.named_modules():
#         #         if isinstance(module, _BatchNorm):
#         #             name = name.replace('_', '.')
#         #             # self.task_bn[i] = deepcopy(module)
#         #             setattr(self.model, name, bns[name])
#         #
#         # return self.model(x=x, task_labels=task_labels)
