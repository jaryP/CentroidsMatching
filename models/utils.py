from typing import Union, Callable, Tuple, Dict

import torch
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.utils.dataset_utils import ConstantSequence
from avalanche.models import DynamicModule, MultiTaskModule
from avalanche.models.helper_method import MultiTaskDecorator
from torch import nn


class CustomMultiHeadClassifier(MultiTaskModule):
    def __init__(self, in_features, heads_generator, out_features=None):
        super().__init__()

        self.heads_generator = heads_generator
        self.in_features = in_features
        self.starting_out_features = out_features
        self.classifiers = torch.nn.ModuleDict()

        # needs to create the first head because pytorch optimizers
        # fail when model.parameters() is empty.
        # first_head = heads_generator(self.in_features,
        #                              self.starting_out_features)
        # self.classifiers['0'] = first_head

    def adaptation(self, dataset: AvalancheDataset):
        super().adaptation(dataset)
        task_labels = dataset.targets_task_labels
        if isinstance(task_labels, ConstantSequence):
            # task label is unique. Don't check duplicates.
            task_labels = [task_labels[0]]

        for tid in set(task_labels):
            tid = str(tid)  # need str keys
            if tid not in self.classifiers:

                if self.starting_out_features is None:
                    out = max(dataset.targets) + 1
                else:
                    out = self.starting_out_features

                new_head = self.heads_generator(self.in_features, out)
                self.classifiers[tid] = new_head

    def forward_single_task(self, x, task_label):
        return self.classifiers[str(task_label)](x)


class CustomMultiTaskDecorator(MultiTaskModule):
    def __init__(self, model: nn.Module, classifier_name: str, heads_generator,
                 out_features=None):
        self.__dict__['_initialized'] = False
        super().__init__()
        self.model = model
        self.classifier_name = classifier_name

        old_classifier = getattr(model, classifier_name)

        if isinstance(old_classifier, nn.Linear):
            in_size = old_classifier.in_features
            out_size = old_classifier.out_features
            old_params = [torch.clone(p.data) for p in
                          old_classifier.parameters()]
            # Replace old classifier by empty block
            setattr(self.model, classifier_name, nn.Sequential())
        elif isinstance(old_classifier, nn.Sequential):
            in_size = old_classifier[-1].in_features
            out_size = old_classifier[-1].out_features
            old_params = [torch.clone(p.data) for p in
                          old_classifier[-1].parameters()]
            del old_classifier[-1]
        else:
            raise NotImplementedError(f"Cannot handle the following type \
            of classification layer {type(old_classifier)}")

        # Set new classifier and initialize to previous param values
        setattr(self, classifier_name,
                CustomMultiHeadClassifier(in_size, heads_generator,
                                          out_features))

        # for param, param_old in \
        #         zip(getattr(self, classifier_name).parameters(), old_params):
        #     param.data = param_old

        self._initialized = True

    def forward_single_task(self, x: torch.Tensor, task_label: int):
        out = self.model(x)
        return getattr(self, self.classifier_name)(out.view(out.size(0), -1),
                                                   task_labels=task_label)

    def __getattr__(self, name):
        # Override pytorch impl from nn.Module

        # Its a bit particular since pytorch nn.Module does not
        # keep some attributes in a classical manner in self.__dict__
        # rather it puts them into _parameters, _buffers and
        # _modules attributes. We have to add these lines to avoid recursion
        if name == 'model':
            return self.__dict__['_modules']['model']
        if name == self.classifier_name:
            return self.__dict__['_modules'][self.classifier_name]

        # If its a different attribute, return the one from the model
        return getattr(self.model, name)

    def __setattr__(self, name, value):
        # During initialization, use pytorch routine
        if not self.__dict__['_initialized'] or name in self.__dict__:
            super().__setattr__(name, value)
        else:
            return setattr(self.model, name, value)


class MultiHeadBackbone(MultiTaskModule):
    def __init__(self,
                 backbone: nn.Module,
                 backbone_output_size: Union[int, tuple],
                 incremental_classifier_f: Callable[[Union[int, tuple],
                                                     int],
                                                    nn.Module]):

        super().__init__()

        self.backbone = backbone
        self.head_in_feat = backbone_output_size

        self.incremental_classifier_f = incremental_classifier_f
        self.heads = dict()

    def adaptation(self, dataset: AvalancheDataset = None):
        task_labels = set(dataset.targets)
        if isinstance(task_labels, ConstantSequence):
            task_labels = [task_labels[0]]

        # for tid in set(task_labels):
        #     tid = str(tid)  # need str keys
        #     if tid not in self.heads:
        new_head = self.incremental_classifier_f(self.head_in_feat,
                                                 len(task_labels))
        self.heads[dataset.targets_task_labels[0]] = new_head

    def forward_single_task(self, x: torch.Tensor,
                            task_label: int,
                            return_embs: bool = False) \
            -> Tuple[torch.Tensor, torch.Tensor]:

        """ compute the output given the input `x` and task label.

        :param x:
        :param task_label: a single task label.
        :return:
        """
        e = self.backbone(x)
        o = self.heads[task_label](e)

        if return_embs:
            return e, o
        return o

    def forward_all_tasks(self, x: torch.Tensor, return_embs: bool = False):
        """ compute the output given the input `x` and task label.
        By default, it considers only tasks seen at training time.

        :param x:
        :return: all the possible outputs are returned as a dictionary
            with task IDs as keys and the output of the corresponding
            task as output.
        """
        res = {}
        for task_id in self.known_train_tasks_labels:
            if return_embs:
                res[task_id] = self.forward_single_task(x, task_id)
            else:
                res[task_id] = self.forward_single_task(x, task_id)[1]
        return res

    def forward(self, x: torch.Tensor, task_labels: torch.Tensor = None,
                return_embs: bool = False) \
            -> Union[Tuple[torch.Tensor, torch.Tensor],
                     Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:

        """ compute the output given the input `x` and task labels.

        :param x:
        :param task_labels: task labels for each sample. if None, the
            computation will return all the possible outputs as a dictionary
            with task IDs as keys and the output of the corresponding task as
            output.
        :return:
        """

        if task_labels is None:
            return self.forward_all_tasks(x, return_embs)

        tasks = list(set(task_labels.tolist()))
        if len(tasks) == 1:
            # fast path. mini-batch is single task.
            return self.forward_single_task(x, tasks[0], return_embs)
        else:
            unique_tasks = torch.unique(task_labels)

        res = {}

        for task in unique_tasks:
            task_mask = task_labels == task
            x_task = x[task_mask]
            out = self.forward_single_task(x_task,
                                           task.item(),
                                           return_embs)
            if not return_embs:
                res[task] = out[:, 1]
            else:
                res[task.item()] = out

        return res


class EmbeddingModelDecorator(MultiTaskModule):
    def __init__(self, model: MultiTaskDecorator):
        super().__init__()

        self.wrapped_class = model
        self.classifier_name = model.classifier_name

    # def adaptation(self, dataset: AvalancheDataset = None):
    #     self.wrapped_class.adaptation(dataset)

    def forward_single_task(self, x: torch.Tensor, task_label: int,
                            return_embeddings: bool = False):
        out = self.wrapped_class.model(x)
        logits = getattr(self.wrapped_class, self.classifier_name) \
            (out.view(out.size(0), -1), task_labels=task_label)

        if return_embeddings:
            return out, logits

        return logits

    def forward_all_tasks(self, x: torch.Tensor,
                          return_embeddings: bool = False):

        """
         compute the output given the input `x` and task label.
        By default, it considers only tasks seen at training time.

        :param x:
        :return: all the possible outputs are returned as a dictionary
            with task IDs as keys and the output of the corresponding
            task as output.
        """

        res = {}
        for task_id in self.known_train_tasks_labels:
            res[task_id] = self.forward_single_task(x,
                                                    task_id,
                                                    return_embeddings)
        return res

    def forward(self, x: torch.Tensor, task_labels: torch.Tensor,
                return_embeddings: bool = False) \
            -> torch.Tensor:
        """ compute the output given the input `x` and task labels.

        :param x:
        :param task_labels: task labels for each sample. if None, the
            computation will return all the possible outputs as a dictionary
            with task IDs as keys and the output of the corresponding task as
            output.
        :return:
        """
        if task_labels is None:
            return self.forward_all_tasks(x,
                                          return_embeddings=return_embeddings)

        if isinstance(task_labels, int):
            # fast path. mini-batch is single task.
            return self.forward_single_task(x, task_labels, return_embeddings)
        else:
            unique_tasks = torch.unique(task_labels)

        out = None
        for task in unique_tasks:
            task_mask = task_labels == task
            x_task = x[task_mask]
            out_task = self.forward_single_task(x_task, task.item(),
                                                return_embeddings)

            if out is None:
                out = torch.empty(x.shape[0], *out_task.shape[1:],
                                  device=out_task.device)
            out[task_mask] = out_task
        return out


class CombinedModel(MultiTaskModule):
    def __init__(self, backbone: nn.Module, classifier: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward_single_task(self, x: torch.Tensor, task_label: int,
                            return_embeddings: bool = False):
        out = self.backbone(x)
        out = torch.flatten(out, 1)
        logits = self.classifier(out, task_labels=task_label)

        if return_embeddings:
            return out, logits

        return logits

    def forward_all_tasks(self, x: torch.Tensor,
                          return_embeddings: bool = False):

        res = {}
        for task_id in self.known_train_tasks_labels:
            res[task_id] = self.forward_single_task(x,
                                                    task_id,
                                                    return_embeddings)
        return res

    def forward(self, x: torch.Tensor, task_labels: torch.Tensor,
                return_embeddings: bool = False) \
            -> torch.Tensor:

        if task_labels is None:
            return self.forward_all_tasks(x,
                                          return_embeddings=return_embeddings)

        if isinstance(task_labels, int):
            # fast path. mini-batch is single task.
            return self.forward_single_task(x, task_labels, return_embeddings)
        else:
            unique_tasks = torch.unique(task_labels)

        out = None
        for task in unique_tasks:
            task_mask = task_labels == task
            x_task = x[task_mask]
            out_task = self.forward_single_task(x_task, task.item(),
                                                return_embeddings)

            if out is None:
                out = torch.empty(x.shape[0], *out_task.shape[1:],
                                  device=out_task.device)
            out[task_mask] = out_task
        return out


# class CombinedModel(nn.Module):
#     def __init__(self, backbone: nn.Module, classifier: nn.Module):
#         super().__init__()
#         self.backbone = backbone
#         self.classifier = classifier
#
#     def forward(self, x):
#         return self.classifier(torch.flatten(self.backbone(x), 1))
