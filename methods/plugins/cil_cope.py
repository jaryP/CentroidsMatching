import torch
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.training import BaseStrategy, ClassBalancedBuffer
from avalanche.training.plugins import StrategyPlugin
from methods.plugins.cope_utils import PrototypeScheme, PPPloss


class ContinualPrototypeEvolution(StrategyPlugin):
    def __init__(self,
                 p_mode='batch_momentum_incr',
                 p_momentum=0.9,
                 n_iter=1,
                 memory_size: int = 200,
                 **kwargs):

        super().__init__()

        self.n_iter = n_iter

        self.pp_scheme = PrototypeScheme(p_mode, p_momentum=p_momentum)
        self.ppp_loss = PPPloss()

        self.memory = ClassBalancedBuffer(max_size=memory_size)

        self.prototypes = None

        # if sit:
        #
        #     if memory_type == 'random':
        #         self.storage_policy = RandomMemory(**memory_parameters)
        #     elif memory_type == 'clustering':
        #         self.storage_policy = ClusteringMemory(**memory_parameters)
        #     else:
        #         assert False, 'Unknown memory type '
        #
        #     if merging_strategy == 'scale_translate':
        #         self.scaler = ScaleTranslate()
        #     elif merging_strategy == 'none':
        #         self.scaler = FakeMerging()
        #     else:
        #         self.scaler = Projector(merging_strategy)
        #
        #     if centroids_merging_strategy is not None:
        #         if merging_strategy == 'scale_translate':
        #             self.centroids_scaler = ScaleTranslate()
        #         elif merging_strategy == 'none':
        #
        #             self.centroids_scaler = FakeMerging()
        #         else:
        #             self.centroids_scaler = Projector(merging_strategy)
        #     else:
        #         self.centroids_scaler = None

    def before_training_exp(self, strategy: "BaseStrategy",
                            num_workers: int = 0, shuffle: bool = True,
                            **kwargs):

        if len(self.memory.buffer_datasets) == 0:
            return

        strategy.dataloader = ReplayDataLoader(
            strategy.adapted_dataset,
            self.memory.buffer,
            oversample_small_tasks=True,
            num_workers=num_workers,
            batch_size=strategy.train_mb_size,
            shuffle=shuffle)

    def loss(self, strategy, **kwargs):
        if not strategy.model.training:
            return -1

        mb_output, y, x = strategy.mb_output, strategy.mb_y, strategy.mb_x

        # if self.prototypes is None:
        #     classes = set(
        #         strategy.experience.benchmark.original_train_dataset.targets)
        #     n_classes = len(classes)
        #
        #     self.pp_scheme.initialize_prototypes(mb_output, classes)

        #     out = torch.full((mb_output.shape[-1], n_classes),
        #                      fill_value=1.0 / n_classes,
        #                      device=strategy.device)
        #     # out.fill_(1.0 / n_classes)
        #     self.prototypes = out
        #
        # tid = strategy.experience.current_experience
        # tasks = strategy.mb_task_id

        loss = self.ppp_loss(x_metric=mb_output,
                             labels=y,
                             prototypes=self.pp_scheme.get_prototypes())

        return loss

    def after_forward(self, strategy: 'BaseStrategy', **kwargs):

        self.pp_scheme.initialize_prototypes(strategy.mb_output,
                                             strategy.experience.classes_in_this_experience)

        tid = strategy.experience.task_label
        mask = strategy.mb_task_id == tid

        self.pp_scheme(
            f=strategy.mb_output,
            y=strategy.mb_y,
            replay_mask=~mask,
            pre_loss=True)

    def after_training_exp(self, strategy: 'BaseStrategy', **kwargs):
        self.memory.update(strategy)

        tid = strategy.experience.current_experience
        mask = strategy.mb_task_id == tid

        self.pp_scheme(
            f=strategy.mb_output,
            y=strategy.mb_y,
            replay_mask=~mask,
            pre_loss=True)

    @torch.no_grad()
    def calculate_classes(self, strategy, embeddings):
        x = embeddings
        c, cy = self.pp_scheme.get_prototypes()

        similarity = torch.einsum('id,jd->ij', x, c)

        pred = torch.argmax(similarity, 1)
        pred = torch.index_select(cy, 0, pred)

        return pred
