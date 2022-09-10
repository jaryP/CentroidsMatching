from avalanche.training import Cumulative, GEM, Replay, Naive, JointTraining, \
    EWC, ICaRL
from avalanche.training.plugins import GEMPlugin, ReplayPlugin

from methods.strategies import EmbeddingRegularization, \
    ContinualMetricLearning, CustomEWC, SeparatedSoftmaxIncrementalLearning, \
    CoPE, MemoryContinualMetricLearning

from models.utils import CombinedModel


def get_plugin(name, **kwargs):
    name = name.lower()
    if name == 'gem':
        return GEMPlugin(
            patterns_per_experience=kwargs['patterns_per_experience'],
            memory_strength=kwargs.get('patterns_per_exp', 0.5))
    elif name == 'none':
        return None
    elif name == 'replay':
        return ReplayPlugin(mem_size=kwargs['mem_size'])
    else:
        assert False


def get_trainer(name, tasks, sit: bool = False, **kwargs):
    name = name.lower()
    num_experiences = len(tasks.train_stream)

    def f(model: CombinedModel,
          criterion, optimizer,
          train_epochs: int,
          train_mb_size: int,
          evaluator,
          device):

        if name == 'gem':
            return GEM(patterns_per_exp=kwargs['patterns_per_experience'],
                       memory_strength=kwargs.get('memory_strength', 0.5),
                       model=model, criterion=criterion, optimizer=optimizer,
                       train_epochs=train_epochs, train_mb_size=train_mb_size,
                       evaluator=evaluator, device=device)
        elif name == 'ewc':
            return CustomEWC(ewc_lambda=kwargs['ewc_lambda'],
                             mode='separate',
                             decay_factor=kwargs.get('decay_factor', None),
                             keep_importance_data=kwargs.get(
                                 'keep_importance_data',
                                 False),
                             model=model, criterion=criterion,
                             optimizer=optimizer,
                             train_epochs=train_epochs,
                             train_mb_size=train_mb_size,
                             evaluator=evaluator, device=device)
        elif name == 'oewc':
            return CustomEWC(ewc_lambda=kwargs['ewc_lambda'],
                             mode='online',
                             decay_factor=kwargs.get('decay_factor', 1),
                             keep_importance_data=kwargs.get(
                                 'keep_importance_data',
                                 False),
                             model=model, criterion=criterion,
                             optimizer=optimizer,
                             train_epochs=train_epochs,
                             train_mb_size=train_mb_size,
                             evaluator=evaluator, device=device)
        elif name == 'replay':
            return Replay(mem_size=kwargs['mem_size'], model=model,
                          criterion=criterion, optimizer=optimizer,
                          train_epochs=train_epochs,
                          train_mb_size=train_mb_size, evaluator=evaluator,
                          device=device)
        elif name == 'cumulative':
            return Cumulative(model=model, criterion=criterion,
                              optimizer=optimizer, train_epochs=train_epochs,
                              train_mb_size=train_mb_size, evaluator=evaluator,
                              device=device)
        elif name == 'naive' or name == 'none':
            return Naive(model=model, criterion=criterion, optimizer=optimizer,
                         train_epochs=train_epochs, train_mb_size=train_mb_size,
                         evaluator=evaluator, device=device)
        elif name == 'joint':
            return JointTraining(model=model, criterion=criterion,
                                 optimizer=optimizer, train_epochs=train_epochs,
                                 train_mb_size=train_mb_size,
                                 evaluator=evaluator, device=device)
        elif name == 'er':
            return EmbeddingRegularization(mem_size=kwargs['mem_size'],
                                           penalty_weight=kwargs.get(
                                               'penalty_weight', 1),
                                           model=model, criterion=criterion,
                                           optimizer=optimizer,
                                           train_epochs=train_epochs,
                                           train_mb_size=train_mb_size,
                                           evaluator=evaluator, device=device,
                                           feature_extractor=model.feature_extractor,
                                           classifier=model.classifier)
        elif name == 'cml':
            return ContinualMetricLearning(model=model,
                                           dev_split_size=kwargs.
                                           get('dev_split_size', 100),
                                           penalty_weight=kwargs.
                                           get('penalty_weight', 1),
                                           sit_memory_size=kwargs.
                                           get('sit_memory_size', 500),
                                           proj_w=kwargs.get('proj_w', 1),
                                           merging_strategy=kwargs.get(
                                               'merging_strategy',
                                               'scale_translate'),
                                           memory_parameters=kwargs.get(
                                               'memory_parameters', {}),
                                           memory_type=kwargs.get('memory_type',
                                                                  'random'),
                                           centroids_merging_strategy=kwargs.get(
                                               'centroids_merging_strategy',
                                               'None'),
                                           num_experiences=num_experiences,
                                           optimizer=optimizer,
                                           criterion=criterion,
                                           train_mb_size=train_mb_size,
                                           train_epochs=train_epochs,
                                           device=device,
                                           sit=sit,
                                           evaluator=evaluator,
                                           eval_every=-1)
        elif name == 'mcml':
            return MemoryContinualMetricLearning(model=model,
                                                 dev_split_size=kwargs.
                                                 get('dev_split_size', 100),
                                                 memory_size=kwargs.
                                                 get('memory_size', 500),
                                                 optimizer=optimizer,
                                                 criterion=criterion,
                                                 train_mb_size=train_mb_size,
                                                 train_epochs=train_epochs,
                                                 device=device,
                                                 sit=sit,
                                                 evaluator=evaluator,
                                                 eval_every=-1)
        elif name == 'icarl':
            return ICaRL(feature_extractor=model.feature_extractor,
                         classifier=model.classifier,
                         memory_size=kwargs.get('memory_size'),
                         buffer_transform=kwargs.get('buffer_transform', None),
                         fixed_memory=kwargs.get('fixed_memory', True),
                         optimizer=optimizer,
                         criterion=criterion,
                         train_mb_size=train_mb_size,
                         train_epochs=train_epochs,
                         device=device,
                         evaluator=evaluator)
        elif name == 'ssil':
            return SeparatedSoftmaxIncrementalLearning(
                model=model,
                sit_memory_size=kwargs.get('memory_size'),
                optimizer=optimizer,
                criterion=criterion,
                train_mb_size=train_mb_size,
                train_epochs=train_epochs,
                device=device,
                evaluator=evaluator
            )
        elif name == 'cope':
            return CoPE(memory_size=kwargs['memory_size'],
                        model=model, criterion=criterion, optimizer=optimizer,
                        train_epochs=train_epochs, train_mb_size=train_mb_size,
                        evaluator=evaluator, device=device)
        else:
            assert False, f'CL method not found {name.lower()}'

    return f
