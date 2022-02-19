from typing import Optional, Any

import avalanche
from avalanche.benchmarks import nc_benchmark

import utils


def nc_scenario(name: str,
                n_tasks: int,
                return_task_id: bool,
                shuffle: bool = True,
                train_transform: Optional[Any] = None,
                eval_transform: Optional[Any] = None,
                seed: Optional[int] = None):
    name = name.lower()

    for v in dir(avalanche.benchmarks):
        if v.lower() == name:
            return getattr(avalanche.benchmarks, v)(n_experiences=n_tasks,
                                                    return_task_id=return_task_id,
                                                    shuffle=shuffle,
                                                    # train_transform=train_transform,
                                                    # eval_transform=eval_transform,
                                                    seed=seed)

    return None


def get_scenario(train_dataset,
                 test_dataset,
                 n_tasks: int,
                 return_task_id: bool,
                 shuffle: bool = True,
                 train_transform: Optional[Any] = None,
                 eval_transform: Optional[Any] = None,
                 seed: Optional[int] = None):

    if return_task_id:
        return nc_benchmark(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            n_experiences=n_tasks,
            task_labels=True,
            seed=seed,
            fixed_class_order=None,
            shuffle=shuffle,
            class_ids_from_zero_in_each_exp=True,
            train_transform=train_transform,
            eval_transform=eval_transform)
    else:
        return nc_benchmark(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            n_experiences=n_tasks,
            task_labels=False,
            seed=seed,
            fixed_class_order=None,
            shuffle=shuffle,
            train_transform=train_transform,
            eval_transform=eval_transform)


def get_dataset_by_name(name: str, root: str = None):
    name = name.lower()

    f = getattr(utils, name, None)
    if f is None:
        return None

    train_split, test_split, train_t, test_t = f(root)

    return train_split, test_split, train_t, test_t


def get_dataset_nc_scenario(name: str,
                            n_tasks: int,
                            return_task_id: bool,
                            shuffle: bool = True,
                            seed: Optional[int] = None):
    name = name.lower()

    avalanche_scenario = nc_scenario(name=name,
                                     n_tasks=n_tasks,
                                     return_task_id=return_task_id,
                                     shuffle=shuffle,
                                     train_transform=None,
                                     eval_transform=None,
                                     seed=seed)

    if avalanche_scenario is not None:
        return avalanche_scenario

    r = get_dataset_by_name(name)
    if r is None:
        assert False, f'Dataset {name} not found.'

    train_split, test_split, train_t, test_t = r

    return get_scenario(train_dataset=train_split, train_transform=train_t,
                        test_dataset=test_split, eval_transform=test_t,
                        n_tasks=n_tasks, return_task_id=return_task_id,
                        shuffle=shuffle, seed=seed)
