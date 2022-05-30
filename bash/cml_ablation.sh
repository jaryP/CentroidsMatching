#!/usr/bin/env bash

DEVICE=$1

#python main.py +scenario=task_incremental_cifar10 +model=resnet20 experiment.experiments=1 +training=cifar10 training.epochs=20 +method=cml_01 +method.dev_split_size=10 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ablation/ti_cifar10/resnet20/cml/cml_01_dev10'
#python main.py +scenario=task_incremental_cifar10 +model=resnet20 experiment.experiments=1 +training=cifar10 training.epochs=20 +method=cml_01 +method.dev_split_size=50 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ablation/ti_cifar10/resnet20/cml/cml_01_dev50'
#python main.py +scenario=task_incremental_cifar10 +model=resnet20 experiment.experiments=1 +training=cifar10 training.epochs=20 +method=cml_01 +method.dev_split_size=100 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ablation/ti_cifar10/resnet20/cml/cml_01_dev100'
#python main.py +scenario=task_incremental_cifar10 +model=resnet20 experiment.experiments=1 +training=cifar10 training.epochs=20 +method=cml_01 +method.dev_split_size=200 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ablation/ti_cifar10/resnet20/cml/cml_01_dev200'

python main.py +scenario=class_incremental_cifar10 +model=resnet20 experiment.experiments=1 +training=cifar10 training.epochs=20 +method=cml_01 +method.dev_split_size=0.1 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ablation/ci_cifar10/resnet20/cml/cml_01_dev01'
python main.py +scenario=class_incremental_cifar10 +model=resnet20 experiment.experiments=1 +training=cifar10 training.epochs=20 +method=cml_01 +method.dev_split_size=0.05 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ablation/ci_cifar10/resnet20/cml/cml_01_dev005'
python main.py +scenario=class_incremental_cifar10 +model=resnet20 experiment.experiments=1 +training=cifar10 training.epochs=20 +method=cml_01 +method.dev_split_size=100 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ablation/ci_cifar10/resnet20/cml/cml_01_dev100'
python main.py +scenario=class_incremental_cifar10 +model=resnet20 experiment.experiments=1 +training=cifar10 training.epochs=20 +method=cml_01 +method.dev_split_size=200 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ablation/ci_cifar10/resnet20/cml/cml_01_dev200'
python main.py +scenario=class_incremental_cifar10 +model=resnet20 experiment.experiments=1 +training=cifar10 training.epochs=20 +method=cml_01 +method.dev_split_size=50 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ablation/ci_cifar10/resnet20/cml/cml_01_dev50'
python main.py +scenario=class_incremental_cifar10 +model=resnet20 experiment.experiments=1 +training=cifar10 training.epochs=20 +method=cml_01 +method.dev_split_size=10 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ablation/ci_cifar10/resnet20/cml/cml_01_dev10'
