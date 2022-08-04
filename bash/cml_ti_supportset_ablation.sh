#!/usr/bin/env bash

DEVICE=$1

python main.py +scenario=task_incremental_cifar10 +model=resnet20  +training=cifar10  +method=cml_01 +method.dev_split_size=10 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ablation/ti_cifar10/resnet20/cml/cml_01_dev10' experiment.experiments=2
python main.py +scenario=task_incremental_cifar10 +model=resnet20  +training=cifar10  +method=cml_01 +method.dev_split_size=50 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ablation/ti_cifar10/resnet20/cml/cml_01_dev50' experiment.experiments=2
python main.py +scenario=task_incremental_cifar10 +model=resnet20  +training=cifar10  +method=cml_01 +method.dev_split_size=100 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ablation/ti_cifar10/resnet20/cml/cml_01_dev100' experiment.experiments=2
python main.py +scenario=task_incremental_cifar10 +model=resnet20  +training=cifar10  +method=cml_01 +method.dev_split_size=200 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ablation/ti_cifar10/resnet20/cml/cml_01_dev200' experiment.experiments=2
python main.py +scenario=task_incremental_cifar10 +model=resnet20  +training=cifar10  +method=cml_01 +method.dev_split_size=500 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ablation/ti_cifar10/resnet20/cml/cml_01_dev500' experiment.experiments=2


python main.py +scenario=class_incremental_cifar10 +model=resnet20  +training=cifar10  +method=cml_01 +method.dev_split_size=10 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ablation/ci_cifar10/resnet20/cml/cml_01_dev10' experiment.experiments=2
python main.py +scenario=class_incremental_cifar10 +model=resnet20  +training=cifar10  +method=cml_01 +method.dev_split_size=50 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ablation/ci_cifar10/resnet20/cml/cml_01_dev50' experiment.experiments=2
python main.py +scenario=class_incremental_cifar10 +model=resnet20  +training=cifar10  +method=cml_01 +method.dev_split_size=100 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ablation/ci_cifar10/resnet20/cml/cml_01_dev100' experiment.experiments=2
python main.py +scenario=class_incremental_cifar10 +model=resnet20  +training=cifar10  +method=cml_01 +method.dev_split_size=200 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ablation/ci_cifar10/resnet20/cml/cml_01_dev200' experiment.experiments=2
python main.py +scenario=class_incremental_cifar10 +model=resnet20  +training=cifar10  +method=cml_01 +method.dev_split_size=500 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ablation/ci_cifar10/resnet20/cml/cml_01_dev500' experiment.experiments=2


#python main.py +scenario=task_incremental_cifar100 +model=resnet20  +training=cifar10  +method=cml_01 +method.dev_split_size=10 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ablation/ti_cifar100/resnet20/cml/cml_01_dev10'  experiment.experiments=1
python main.py +scenario=task_incremental_cifar100 +model=resnet20  +training=cifar10  +method=cml_01 +method.dev_split_size=50 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ablation/ti_cifar100/resnet20/cml/cml_01_dev50'  experiment.experiments=1
python main.py +scenario=task_incremental_cifar100 +model=resnet20  +training=cifar10 +method=cml_01 +method.dev_split_size=100 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ablation/ti_cifar100/resnet20/cml/cml_01_dev100'  experiment.experiments=1
python main.py +scenario=task_incremental_cifar100 +model=resnet20  +training=cifar10  +method=cml_01 +method.dev_split_size=200 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ablation/ti_cifar100/resnet20/cml/cml_01_dev200'  experiment.experiments=1
python main.py +scenario=task_incremental_cifar100 +model=resnet20  +training=cifar10  +method=cml_01 +method.dev_split_size=500 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ablation/ti_cifar100/resnet20/cml/cml_01_dev500'  experiment.experiments=1
