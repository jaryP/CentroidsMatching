from avalanche.training import Cumulative, GEM, Replay, Naive, JointTraining, \
    EWC, ICaRL
from avalanche.training.plugins import GEMPlugin, ReplayPlugin

from methods.strategies import EmbeddingRegularization, ContinualMetricLearning
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
    # elif name == 'cumulative':
    #     return Cumulative()
    # elif name == 'naive':
    #     Naive
    else:
        assert False


def get_trainer(name, sit: bool = False, **kwargs):
    name = name.lower()

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
            return EWC(ewc_lambda=kwargs['ewc_lambda'],
                       mode=kwargs.get('mode', 'separate'),
                       decay_factor=kwargs.get('decay_factor', None),
                       keep_importance_data=kwargs.get('keep_importance_data',
                                                       False),
                       model=model, criterion=criterion, optimizer=optimizer,
                       train_epochs=train_epochs, train_mb_size=train_mb_size,
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
                                           feature_extractor=model.backbone,
                                           classifier=model.classifier)
        elif name == 'cml':
            return ContinualMetricLearning(model=model,
                                           dev_split_size=kwargs.
                                           get('dev_split_size', 0.1),
                                           penalty_weight=kwargs.
                                           get('penalty_weight', 1),
                                           optimizer=optimizer,
                                           criterion=criterion,
                                           train_mb_size=train_mb_size,
                                           train_epochs=train_epochs,
                                           device=device,
                                           sit=sit,
                                           evaluator=evaluator)
        elif name == 'icarl':
            return ICaRL(feature_extractor=model.backbone,
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
        else:
            assert False

    return f
