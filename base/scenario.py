from typing import Optional
from avalanche.benchmarks import nc_benchmark

from avalanche.benchmarks.classic.ccifar10 import _get_cifar10_dataset, \
    _default_cifar10_train_transform, _default_cifar10_eval_transform

from avalanche.benchmarks.classic.ccifar100 import _get_cifar100_dataset, \
    _default_cifar100_train_transform, _default_cifar100_eval_transform

from avalanche.benchmarks.classic.ctiny_imagenet import \
    _get_tiny_imagenet_dataset

from avalanche.benchmarks.classic.ctiny_imagenet import \
    _default_train_transform as _default_tiny_imagenet_train_transform
from avalanche.benchmarks.classic.ctiny_imagenet import \
    _default_eval_transform as _default_tiny_imagenet_eval_transform


def get_dataset_by_name(name: str, root: str = None):
    name = name.lower()

    if name == 'cifar10':
        train_set, test_set = _get_cifar10_dataset(root)
        train_t = _default_cifar10_train_transform
        test_t = _default_cifar10_eval_transform
    elif name == 'cifar100':
        train_set, test_set = _get_cifar100_dataset(root)
        train_t = _default_cifar100_train_transform
        test_t = _default_cifar100_eval_transform
    elif name == 'tinyimagenet':
        train_set, test_set = _get_tiny_imagenet_dataset(root)
        train_t = _default_tiny_imagenet_train_transform
        test_t = _default_tiny_imagenet_eval_transform
    else:
        return None

    return train_set, test_set, train_t, test_t


def get_dataset_nc_scenario(name: str,
                            scenario: str,
                            n_tasks: int,
                            til: bool,
                            shuffle: bool = True,
                            seed: Optional[int] = None,
                            force_sit=False,
                            method_name=None,
                            sit_with_labels=False):

    name = name.lower()

    r = get_dataset_by_name(name)

    if r is None:
        assert False, f'Dataset {name} not found.'

    train_split, test_split, train_t, test_t = r

    if method_name == 'cope':
        return nc_benchmark(
            train_dataset=train_split,
            test_dataset=test_split,
            n_experiences=n_tasks,
            task_labels=True,
            seed=seed,
            fixed_class_order=None,
            shuffle=shuffle,
            class_ids_from_zero_in_each_exp=False,
            class_ids_from_zero_from_first_exp=True,
            train_transform=train_t,
            eval_transform=test_t)

    if til and not force_sit:
        return nc_benchmark(
            train_dataset=train_split,
            test_dataset=test_split,
            n_experiences=n_tasks,
            task_labels=True,
            seed=seed,
            fixed_class_order=None,
            shuffle=shuffle,
            class_ids_from_zero_in_each_exp=True,
            train_transform=train_t,
            eval_transform=test_t)
    else:
        return nc_benchmark(
            train_dataset=train_split,
            test_dataset=test_split,
            n_experiences=n_tasks,
            task_labels=True,
            seed=seed,
            class_ids_from_zero_from_first_exp=True,
            fixed_class_order=None,
            shuffle=shuffle,
            train_transform=train_t,
            eval_transform=test_t)
