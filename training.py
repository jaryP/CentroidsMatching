import csv
import json
import logging
import os
import pickle
import random
import sys
import warnings
from collections import defaultdict
from typing import List

import numpy as np
import torch
from avalanche.evaluation.metric_results import MetricValue
from avalanche.evaluation.metric_utils import phase_and_task, stream_type
from avalanche.evaluation.metrics import accuracy_metrics, bwt_metrics, \
    forgetting_metrics, timing_metrics, disk_usage_metrics
from avalanche.logging import InteractiveLogger, TextLogger, StrategyLogger
from avalanche.logging.text_logging import UNSUPPORTED_TYPES
from avalanche.training import BaseStrategy
from avalanche.training.plugins import EvaluationPlugin
from omegaconf import DictConfig, OmegaConf
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD

from base.methods import get_trainer
from base.scenario import get_dataset_nc_scenario
from models.base import get_cl_model
from utils import get_save_path, get_optimizer


class CustomTextLogger(StrategyLogger):
    def __init__(self, file=sys.stdout):
        super().__init__()
        self.file = file
        self.metric_vals = {}

    def log_single_metric(self, name, value, x_plot) -> None:
        self.metric_vals[name] = (name, x_plot, value)

    def _val_to_str(self, m_val):
        if isinstance(m_val, torch.Tensor):
            return '\n' + str(m_val)
        elif isinstance(m_val, float):
            return f'{m_val:.4f}'
        else:
            return str(m_val)

    def print_current_metrics(self):
        sorted_vals = sorted(self.metric_vals.values(),
                             key=lambda x: x[0])
        for name, x, val in sorted_vals:
            if isinstance(val, UNSUPPORTED_TYPES):
                continue
            val = self._val_to_str(val)
            print(f'\t{name} = {val}', file=self.file, flush=True)

    def before_training_exp(self, strategy: 'BaseStrategy',
                            metric_values: List['MetricValue'], **kwargs):
        super().before_training_exp(strategy, metric_values, **kwargs)
        self._on_exp_start(strategy)

    def before_eval_exp(self, strategy: 'BaseStrategy',
                        metric_values: List['MetricValue'], **kwargs):
        super().before_eval_exp(strategy, metric_values, **kwargs)
        self._on_exp_start(strategy)

    def after_eval_exp(self, strategy: 'BaseStrategy',
                       metric_values: List['MetricValue'], **kwargs):
        super().after_eval_exp(strategy, metric_values, **kwargs)
        exp_id = strategy.experience.current_experience
        task_id = phase_and_task(strategy)[1]
        if task_id is None:
            print(f'> Eval on experience {exp_id} '
                  f'from {stream_type(strategy.experience)} stream ended.',
                  file=self.file, flush=True)
        else:
            print(f'> Eval on experience {exp_id} (Task '
                  f'{task_id}) '
                  f'from {stream_type(strategy.experience)} stream ended.',
                  file=self.file, flush=True)
        self.print_current_metrics()
        self.metric_vals = {}

    def before_training(self, strategy: 'BaseStrategy',
                        metric_values: List['MetricValue'], **kwargs):
        super().before_training(strategy, metric_values, **kwargs)
        print('-- >> Start of training phase << --', file=self.file, flush=True)

    def before_eval(self, strategy: 'BaseStrategy',
                    metric_values: List['MetricValue'], **kwargs):
        super().before_eval(strategy, metric_values, **kwargs)
        print('-- >> Start of eval phase << --', file=self.file, flush=True)

    def after_training(self, strategy: 'BaseStrategy',
                       metric_values: List['MetricValue'], **kwargs):
        super().after_training(strategy, metric_values, **kwargs)
        print('-- >> End of training phase << --', file=self.file, flush=True)

    def after_eval(self, strategy: 'BaseStrategy',
                   metric_values: List['MetricValue'], **kwargs):
        super().after_eval(strategy, metric_values, **kwargs)
        print('-- >> End of eval phase << --', file=self.file, flush=True)
        self.print_current_metrics()
        self.metric_vals = {}

    def _on_exp_start(self, strategy: 'BaseStrategy'):
        action_name = 'training' if strategy.is_training else 'eval'
        exp_id = strategy.experience.current_experience
        task_id = phase_and_task(strategy)[1]
        stream = stream_type(strategy.experience)
        if task_id is None:
            print('-- Starting {} on experience {} from {} stream --'
                  .format(action_name, exp_id, stream),
                  file=self.file,
                  flush=True)
        else:
            print('-- Starting {} on experience {} (Task {}) from {} stream --'
                  .format(action_name, exp_id, task_id, stream),
                  file=self.file,
                  flush=True)


def avalanche_training(cfg: DictConfig):
    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(cfg))

    scenario = cfg['scenario']
    dataset = scenario['dataset']
    scenario_name = scenario['scenario']
    n_tasks = scenario['n_tasks']
    return_task_id = scenario['return_task_id']

    shuffle = scenario['shuffle']
    shuffle_first = scenario.get('shuffle_first', False)

    seed = scenario.get('seed', None)

    model = cfg['model']
    model_name = model['name']

    method = cfg['method']
    plugin_name = method['name'].lower()
    save_name = method['save_name']

    training = cfg['training']
    epochs = training['epochs']
    batch_size = training['batch_size']
    device = training['device']

    experiment = cfg['experiment']
    n_experiments = experiment.get('experiments', 1)
    load = experiment.get('load', True)
    save = experiment.get('save', True)

    optimizer_cfg = cfg['optimizer']
    optimizer_name = optimizer_cfg.get('optimizer', 'sgd')
    lr = optimizer_cfg.get('lr', 1e-1)
    momentum = optimizer_cfg.get('momentum', 0.9)
    weight_decay = optimizer_cfg.get('weight_decay', 0)

    if torch.cuda.is_available() and device != 'cpu':
        torch.cuda.set_device(device)
        device = 'cuda:{}'.format(device)
    else:
        warnings.warn("Device not found, CUDA not available, "
                      "or device set to cpu")

    device = torch.device(device)

    all_results = []

    base_path = os.getcwd()

    for exp_n in range(1, n_experiments + 1):
        log.info(f'Starting experiment {exp_n} (of {n_experiments})')

        seed = exp_n - 1

        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        experiment_path = os.path.join(base_path, str(seed))

        os.makedirs(experiment_path, exist_ok=True)

        # results_path = os.path.join(experiment_path, 'results.pkl')
        results_path = os.path.join(experiment_path, 'results.json')

        sit = not return_task_id
        return_task_id = return_task_id if plugin_name != 'cml' \
            else True
        # return_task_id = False

        # force_sit = sit and plugin_name == 'cml'
        force_sit = False

        if plugin_name in ['icarl', 'er'] and sit:
            assert sit, 'ICarL and ER only work under Class Incremental Scenario'

        tasks = get_dataset_nc_scenario(name=dataset,
                                        scenario=scenario_name,
                                        n_tasks=n_tasks,
                                        shuffle=shuffle if exp_n < n_experiments else shuffle_first,
                                        return_task_id=return_task_id,
                                        seed=seed,
                                        force_sit=force_sit)

        log.info(f'Original classes: {tasks.classes_order_original_ids}')
        log.info(f'Original classes per exp: {tasks.original_classes_in_exp}')

        if load and os.path.exists(results_path):
            log.info(f'Results loaded')
            # with open(results_path, 'rb') as handle:
            #     results = pickle.load(handle)
            with open(results_path) as json_file:
                results = json.load(json_file)
        else:

            img, _, _ = tasks.train_stream[0].dataset[0]

            model = get_cl_model(model_name=model_name,
                                 input_shape=tuple(img.shape),
                                 method_name=plugin_name,
                                 sit=sit)

            file_path = os.path.join(experiment_path, 'results.txt')
            output_file = open(file_path, 'w')

            eval_plugin = EvaluationPlugin(
                accuracy_metrics(minibatch=False, epoch=False, experience=True,
                                 stream=True),
                # loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
                # timing_metrics(epoch=True),
                # forgetting_metrics(experience=True, stream=False),
                # cpu_usage_metrics(experience=True),
                # confusion_matrix_metrics(num_classes=tasks.n_classes, save_image=False, stream=True),
                # disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
                bwt_metrics(experience=True, stream=True),
                loggers=[
                    # TextLogger(output_file),
                    # TextLogger(),
                    CustomTextLogger(),
                    # InteractiveLogger()
                ],
                # benchmark=tasks,
                # strict_checks=True
            )

            # opt = SGD(model.parameters(), lr=0.01, momentum=0.9)

            opt = get_optimizer(parameters=model.parameters(),
                                name=optimizer_name,
                                lr=lr,
                                weight_decay=weight_decay,
                                momentum=momentum)

            criterion = CrossEntropyLoss()

            trainer = get_trainer(**method,
                                  tasks=tasks,
                                  sit=sit)

            strategy = trainer(model=model,
                               criterion=criterion,
                               optimizer=opt,
                               train_epochs=epochs,
                               train_mb_size=batch_size,
                               evaluator=eval_plugin,
                               device=device)

            results = []

            for i, experience in enumerate(tasks.train_stream):
                strategy.train(experiences=experience)
                results.append(strategy.eval(tasks.test_stream[:i + 1]))
                # print(strategy.eval(tasks.train_stream[:i + 1]))

            output_file.close()

        for res in results:
            average_accuracy = []
            for k, v in res.items():
                if k.startswith('Top1_Acc_Stream/eval_phase/test_stream/'):
                    average_accuracy.append(v)
            average_accuracy = np.mean(average_accuracy)
            res['average_accuracy'] = average_accuracy

        for k, v in results[-1].items():
            log.info(f'Metric {k}:  {v}')
            # if k.startswith('Top1_Acc_Stream/eval_phase/test_stream/'):
            #     average_accuracy.append(v)

        # average_accuracy = np.mean(average_accuracy)
        # input(average_accuracy)
        # results[-1]['average_accuracy'] = average_accuracy
        # with open(results_path, 'wb') as handle:
        #     pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(results_path, 'w') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        all_results.append(results)

    # res_path = get_save_path(scenario_name=scenario_name,
    #                          plugin=plugin_name,
    #                          plugin_name=save_name,
    #                          model_name=model_name)
    log.info(f'Average across the experiments.')

    mean_res = defaultdict(list)
    with open(os.path.join(base_path, 'experiments_results.csv'),
              'w', newline='') as csvfile:

        fieldnames = ['experiment'] + list(results[-1].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, r in enumerate(all_results):
            # res = {'experiment': i + 1}
            # res.update(r[-1])
            # writer.writerow(res)

            for k, v in r[-1].items():
                mean_res[k].append(v)

        m = {k: np.mean(v) for k, v in mean_res.items()}
        s = {k: np.std(v) for k, v in mean_res.items()}

        for k, v in results[-1].items():
            _m = m[k]
            _s = s[k]
            log.info(f'Metric {k}: mean: {_m*100:.2f}, std: {_s*100:.2f}')

        # m.update({'experiment': 'mean'})
        # s.update({'experiment': 'std'})
        #
        # writer.writerow(m)
        # writer.writerow(s)


# def continual_metric_learning_training(cfg: DictConfig):
#     log = logging.getLogger(__name__)
#     log.info(OmegaConf.to_yaml(cfg))
#
#     scenario = cfg['scenario']
#     dataset = scenario['dataset']
#     scenario_name = scenario['scenario']
#     n_tasks = scenario['n_tasks']
#     return_task_id = scenario['return_task_id']
#     shuffle = scenario['shuffle']
#     seed = scenario.get('seed', None)
#
#     model = cfg['model']
#     model_name = model['name']
#
#     method = cfg['method']
#     plugin_name = method['name']
#     save_name = method['save_name']
#
#     training = cfg['training']
#     epochs = training['epochs']
#     batch_size = training['batch_size']
#     device = training['device']
#
#     experiment = cfg['experiment']
#     n_experiments = experiment.get('experiments', 1)
#     load = experiment.get('load', True)
#     save = experiment.get('save', True)
#
#     if torch.cuda.is_available() and device != 'cpu':
#         torch.cuda.set_device(device)
#         device = 'cuda:{}'.format(device)
#     else:
#         warnings.warn("Device not found or CUDA not available.")
#     device = torch.device(device)
#
#     all_results = []
#
#     for exp_n in range(1, n_experiments + 1):
#         log.info(f'Starting experiment {exp_n} (of {n_experiments})')
#
#         seed = exp_n - 1
#
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         random.seed(seed)
#
#         experiment_path = get_save_path(scenario_name=scenario_name,
#                                         plugin=plugin_name,
#                                         plugin_name=save_name,
#                                         model_name=model_name,
#                                         exp_n=exp_n,
#                                         sit=not return_task_id)
#
#         os.makedirs(experiment_path, exist_ok=True)
#
#         results_path = os.path.join(experiment_path, 'results.pkl')
#         if load and os.path.exists(results_path):
#             log.info(f'Results loaded')
#             with open(results_path, 'rb') as handle:
#                 results = pickle.load(handle)
#         else:
#             tasks = get_dataset_nc_scenario(name=dataset,
#                                             scenario=scenario_name,
#                                             n_tasks=n_tasks,
#                                             shuffle=shuffle,
#                                             return_task_id=return_task_id,
#                                             seed=seed)
#
#             img, _, _ = tasks.train_stream[0].dataset[0]
#
#             # plugin = get_plugin(**method)
#
#             model = get_cl_model(model_name=model_name,
#                                  input_shape=tuple(img.shape),
#                                  method_name=plugin_name,
#                                  sit=not return_task_id,
#                                  cml_out_features=256)
#
#             file_path = os.path.join(experiment_path, 'results.txt')
#             output_file = open(file_path, 'w')
#
#             eval_plugin = EvaluationPlugin(
#                 accuracy_metrics(minibatch=False, epoch=False, experience=True,
#                                  stream=True),
#                 # loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
#                 timing_metrics(experience=True),
#                 # forgetting_metrics(experience=True, stream=True),
#                 # cpu_usage_metrics(experience=True),
#                 # confusion_matrix_metrics(num_classes=tasks.n_classes, save_image=False, stream=True),
#                 # disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
#                 bwt_metrics(experience=True, stream=True),
#                 loggers=[TextLogger(output_file), TextLogger()],
#                 # benchmark=tasks,
#                 strict_checks=False
#             )
#
#             opt = SGD(model.parameters(), lr=0.01, momentum=0.9)
#             criterion = CrossEntropyLoss()
#
#             trainer = get_trainer(**method)
#             strategy = trainer(model=model,
#                                criterion=criterion,
#                                optimizer=opt,
#                                train_epochs=epochs,
#                                train_mb_size=batch_size,
#                                evaluator=eval_plugin,
#                                device=device)
#
#             # strategy = BaseStrategy(model=model,
#             #                         plugins=[
#             #                             plugin] if plugin is not None else None,
#             #                         criterion=criterion,
#             #                         optimizer=opt,
#             #                         train_epochs=epochs,
#             #                         train_mb_size=batch_size,
#             #                         evaluator=eval_plugin,
#             #                         device=device)
#
#             results = []
#             for i, experience in enumerate(tasks.train_stream):
#                 strategy.train(experiences=experience)
#                 results.append(strategy.eval(tasks.test_stream[i]))
#                 # print(model)
#
#             output_file.close()
#
#         average_accuracy = []
#         # Top1_Acc_Stream / eval_phase / test_stream /
#         for k, v in results[-1].items():
#             if k.startswith('Top1_Acc_Stream/eval_phase/test_stream/'):
#                 average_accuracy.append(v)
#
#         average_accuracy = np.mean(average_accuracy)
#         input(average_accuracy)
#
#         for k, v in results[-1].items():
#             log.info(f'Metric {k}:  {v}')
#
#         with open(results_path, 'wb') as handle:
#             pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#         all_results.append(results)
#
#     res_path = get_save_path(scenario_name=scenario_name,
#                              plugin=plugin_name,
#                              plugin_name=save_name,
#                              model_name=model_name)
#
#     mean_res = defaultdict(list)
#     with open(os.path.join(res_path, 'experiments_results.csv'),
#               'w', newline='') as csvfile:
#
#         fieldnames = ['experiment'] + list(results[-1].keys())
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#
#         for i, r in enumerate(all_results):
#             res = {'experiment': i + 1}
#             res.update(r[-1])
#             writer.writerow(res)
#
#             for k, v in r[-1].items():
#                 mean_res[k].append(v)
#
#         m = {k: np.mean(v) for k, v in mean_res.items()}
#         s = {k: np.std(v) for k, v in mean_res.items()}
#
#         m.update({'experiment': 'mean'})
#         s.update({'experiment': 'std'})
#
#         writer.writerow(m)
#         writer.writerow(s)

