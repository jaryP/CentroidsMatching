import os

from torch import optim


def get_save_path(scenario_name: str,
                  plugin: str,
                  plugin_name: str,
                  model_name: str,
                  exp_n: int = None,
                  sit: bool = False):

    base_path = os.getcwd()
    if exp_n is None:
        return os.path.join(base_path,
                            model_name,
                            scenario_name if not sit else
                            f'sit_{scenario_name}',
                            plugin,
                            plugin_name)

    experiment_path = os.path.join(base_path,
                                   model_name,
                                   scenario_name if not sit else
                                   f'sit_{scenario_name}',
                                   plugin,
                                   plugin_name,
                                   f'exp_{exp_n}')
    return experiment_path


def get_optimizer(parameters,
                  name: str,
                  lr: float,
                  momentum: float = 0.0,
                  weight_decay: float = 0):

    name = name.lower()
    if name == 'adam':
        return optim.Adam(parameters, lr, weight_decay=weight_decay)
    elif name == 'sgd':
        return optim.SGD(parameters, lr, momentum=momentum,
                         weight_decay=weight_decay)
    else:
        raise ValueError('Optimizer must be adam or sgd')
