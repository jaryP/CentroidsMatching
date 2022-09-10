import torch
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.utils.dataset_utils import ConstantSequence
from avalanche.models import MultiTaskModule
from torch import nn


class CustomMultiHeadClassifier(MultiTaskModule):
    def __init__(self, in_features, heads_generator, out_features=None,
                 p=None):

        super().__init__()

        self.heads_generator = heads_generator
        self.in_features = in_features
        self.starting_out_features = out_features
        self.classifiers = torch.nn.ModuleDict()

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

    def forward_single_task(self, x, task_label, **kwargs):
        return self.classifiers[str(task_label)](x, **kwargs)


class CombinedModel(MultiTaskModule):
    def __init__(self, backbone: nn.Module, classifier: nn.Module, p=None):
        super().__init__()
        self.feature_extractor = backbone
        self.classifier = classifier

        # self.dropout = lambda x: x
        # if p is not None:
        #     self.dropout = Dropout(p)

    def forward_single_task(self, x: torch.Tensor, task_label: int,
                            return_embeddings: bool = False,
                            t=None):

        out = self.feature_extractor(x, task_label)
        out = torch.flatten(out, 1)

        # out = self.dropout(out)

        logits = self.classifier(out, task_labels=task_label)

        if return_embeddings:
            return out, logits

        return logits

    def forward_all_tasks(self, x: torch.Tensor,
                          return_embeddings: bool = False,
                          **kwargs):

        res = {}
        for task_id in self.known_train_tasks_labels:
            res[task_id] = self.forward_single_task(x,
                                                    task_id,
                                                    return_embeddings,
                                                    **kwargs)
        return res

    def forward(self, x: torch.Tensor, task_labels: torch.Tensor,
                return_embeddings: bool = False,
                **kwargs) \
            -> torch.Tensor:

        if task_labels is None:
            return self.forward_all_tasks(x,
                                          return_embeddings=return_embeddings,
                                          **kwargs)

        if isinstance(task_labels, int):
            # fast path. mini-batch is single task.
            return self.forward_single_task(x, task_labels, return_embeddings,
                                            **kwargs)

        unique_tasks = torch.unique(task_labels)
        if len(unique_tasks) == 1:
            return self.forward_single_task(x, unique_tasks.item(),
                                            return_embeddings, **kwargs)

        out = None
        for task in unique_tasks:
            task_mask = task_labels == task
            x_task = x[task_mask]
            out_task = self.forward_single_task(x_task, task.item(),
                                                return_embeddings, **kwargs)

            if out is None:
                out = torch.empty(x.shape[0], *out_task.shape[1:],
                                  device=out_task.device)
            out[task_mask] = out_task
        return out
