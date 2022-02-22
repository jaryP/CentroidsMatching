import csv
import logging
import os
import pickle
import random
import warnings

import numpy as np
import torch
from avalanche.evaluation.metrics import accuracy_metrics, bwt_metrics
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin
from omegaconf import DictConfig, OmegaConf
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD

from base.methods import get_trainer
from base.scenario import get_dataset_nc_scenario
from models.base import get_cl_model
from utils import get_save_path


def avalanche_training(cfg: DictConfig):
    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(cfg))

    scenario = cfg['scenario']
    dataset = scenario['dataset']
    scenario_name = scenario['scenario']
    n_tasks = scenario['n_tasks']
    return_task_id = scenario['return_task_id']
    shuffle = scenario['shuffle']
    seed = scenario.get('seed', None)

    model = cfg['model']
    model_name = model['name']

    method = cfg['method']
    plugin_name = method['name']
    save_name = method['save_name']

    training = cfg['training']
    epochs = training['epochs']
    batch_size = training['batch_size']
    device = training['device']

    experiment = cfg['experiment']
    n_experiments = experiment.get('experiments', 1)
    load = experiment.get('load', True)
    save = experiment.get('save', True)

    if torch.cuda.is_available() and device != 'cpu':
        torch.cuda.set_device(device)
        device = 'cuda:{}'.format(device)
    else:
        warnings.warn("Device not found or CUDA not available.")
    device = torch.device(device)

    all_results = []

    for exp_n in range(1, n_experiments + 1):
        log.info(f'Starting experiment {exp_n} (of {n_experiments})')

        seed = exp_n - 1

        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        experiment_path = get_save_path(scenario_name=scenario_name,
                                        plugin=plugin_name,
                                        plugin_name=save_name,
                                        model_name=model_name,
                                        exp_n=exp_n)

        os.makedirs(experiment_path, exist_ok=True)

        results_path = os.path.join(experiment_path, 'results.pkl')
        if load and os.path.exists(results_path):
            log.info(f'Results loaded')
            with open(results_path, 'rb') as handle:
                results = pickle.load(handle)
        else:
            tasks = get_dataset_nc_scenario(name=dataset,
                                            scenario=scenario_name,
                                            n_tasks=n_tasks,
                                            shuffle=shuffle,
                                            return_task_id=return_task_id,
                                            seed=seed)

            img, _, _ = tasks.train_stream[0].dataset[0]

            # plugin = get_plugin(**method)

            model = get_cl_model(model_name=model_name,
                                 input_shape=tuple(img.shape),
                                 method_name=plugin_name)

            file_path = os.path.join(experiment_path, 'results.txt')
            output_file = open(file_path, 'w')

            eval_plugin = EvaluationPlugin(
                accuracy_metrics(minibatch=False, epoch=False, experience=True,
                                 stream=True),
                # loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
                # timing_metrics(epoch=True),
                # forgetting_metrics(experience=True, stream=True),
                # cpu_usage_metrics(experience=True),
                # confusion_matrix_metrics(num_classes=tasks.n_classes, save_image=False, stream=True),
                # disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
                bwt_metrics(experience=True, stream=True),
                loggers=[TextLogger(output_file), TextLogger()],
                benchmark=tasks,
                strict_checks=True
            )

            opt = SGD(model.parameters(), lr=0.01, momentum=0.9)
            criterion = CrossEntropyLoss()

            trainer = get_trainer(**method)
            strategy = trainer(model=model,
                               criterion=criterion,
                               optimizer=opt,
                               train_epochs=epochs,
                               train_mb_size=batch_size,
                               evaluator=eval_plugin,
                               device=device)

            # strategy = BaseStrategy(model=model,
            #                         plugins=[
            #                             plugin] if plugin is not None else None,
            #                         criterion=criterion,
            #                         optimizer=opt,
            #                         train_epochs=epochs,
            #                         train_mb_size=batch_size,
            #                         evaluator=eval_plugin,
            #                         device=device)

            results = []
            for experience in tasks.train_stream:
                strategy.train(experiences=experience)

                results.append(strategy.eval(tasks.test_stream))

            output_file.close()

        for k, v in results[-1].items():
            log.info(f'Metric {k}:  {v}')

        with open(results_path, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        all_results.append(results)

    res_path = get_save_path(scenario_name=scenario_name,
                             plugin=plugin_name,
                             plugin_name=save_name,
                             model_name=model_name)

    with open(os.path.join(res_path, 'experiments_results.csv'),
              'w', newline='') as csvfile:

        fieldnames = ['experiment'] + list(results[-1].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, r in enumerate(all_results):
            res = {'experiment': i + 1}
            res.update(r[-1])
            writer.writerow(res)
