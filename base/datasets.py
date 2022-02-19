from avalanche.benchmarks.datasets import default_dataset_location
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import Normalize, ToTensor, Compose, RandomCrop, \
    RandomHorizontalFlip


def cifar10(dataset_root):
    train_transform = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465),
                  (0.2023, 0.1994, 0.2010))
    ])

    eval_transform = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465),
                  (0.2023, 0.1994, 0.2010))
    ])

    if dataset_root is None:
        dataset_root = default_dataset_location('cifar10')

    train_set = CIFAR10(dataset_root, train=True, download=True)
    test_set = CIFAR10(dataset_root, train=False, download=True)

    return train_set, test_set, \
           train_transform, \
           eval_transform


def mnist(dataset_root):
    train_transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,))
    ])

    eval_transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,))
    ])

    if dataset_root is None:
        dataset_root = default_dataset_location('mnist')

    train_set = MNIST(root=dataset_root,
                      train=True, download=True)

    test_set = MNIST(root=dataset_root,
                     train=False, download=True)

    return train_set, test_set, \
           train_transform, \
           eval_transform
