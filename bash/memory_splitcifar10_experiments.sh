#!/usr/bin/env bash

DEVICE=$1

#case $METHOD in
#gem)
#python main.py +scenario=task_incremental_cifar10 experiment.experiments=1 +model=resnet20 +training=cifar10 +method=gem_100 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar10/resnet20/gem/gem_100'
#python main.py +scenario=task_incremental_cifar10 experiment.experiments=1 +model=resnet20 +training=cifar10 +method=gem_200 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar10/resnet20/gem/gem_200'
#python main.py +scenario=task_incremental_cifar10 experiment.experiments=1 +model=resnet20 +training=cifar10 +method=gem_500 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar10/resnet20/gem/gem_500'
#python main.py +scenario=task_incremental_cifar10 experiment.experiments=1 +model=resnet20 +training=cifar10 +method=gem_1000 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar10/resnet20/gem/gem_1000'
#;;
#er)
python main.py +scenario=task_incremental_cifar10 experiment.experiments=1 +model=resnet20 +training=cifar10 +method=er_100_05 optimizer=adam  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar10/resnet20/er/er_100_05'
python main.py +scenario=task_incremental_cifar10 experiment.experiments=1 +model=resnet20 +training=cifar10 +method=er_50_05 optimizer=adam  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar10/resnet20/er/er_50_05'
#python main.py +scenario=task_incremental_cifar10 experiment.experiments=1 +model=resnet20 +training=cifar10 +method=er_200 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar10/resnet20/er/er_200'
python main.py +scenario=task_incremental_cifar10 experiment.experiments=1 +model=resnet20 +training=cifar10 +method=er_200_05 optimizer=adam  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar10/resnet20/er/er_200_05'
#python main.py +scenario=task_incremental_cifar10 experiment.experiments=1 +model=resnet20 +training=cifar10 +method=er_200_100 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar10/resnet20/er/er_200_100'
#python main.py +scenario=task_incremental_cifar10 experiment.experiments=1 +model=resnet20 +training=cifar10 +method=er_500_1 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar10/resnet20/er/er_500'
python main.py +scenario=task_incremental_cifar10 experiment.experiments=1 +model=resnet20 +training=cifar10 +method=er_500_05 optimizer=adam  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar10/resnet20/er/er_500_05'
#;;
#cml)
#  python main.py +scenario=task_incremental_cifar10 experiment.experiments=1 +model=resnet20 +training=cifar10 training.epochs=20 +method=cml_05 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar10/resnet20/cml/cml_05'
#  python main.py +scenario=task_incremental_cifar10 experiment.experiments=1 +model=resnet20 +training=cifar10 training.epochs=20 +method=cml_1 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar10/resnet20/cml/cml_1'
#  python main.py +scenario=task_incremental_cifar10 experiment.experiments=1 +model=resnet20 +training=cifar10 training.epochs=20 +method=cml_01 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar10/resnet20/cml/cml_01'
##  python main.py +scenario=task_incremental_cifar10 experiment.experiments=1 +model=resnet20 +training=cifar10 training.epochs=20 +method=cml_10 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar10/resnet20/cml/cml_10'
#;;
#icarl)
#python main.py +scenario=task_incremental_cifar10 experiment.experiments=1 +model=resnet20 +training=cifar10 +method=icarl_500 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar10/resnet20/icarl/icarl_500/'
#python main.py +scenario=task_incremental_cifar10 experiment.experiments=1 +model=resnet20 +training=cifar10 +method=icarl_1000 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar10/resnet20/icarl/icarl_1000/'
#python main.py +scenario=task_incremental_cifar10 experiment.experiments=1 +model=resnet20 +training=cifar10 +method=icarl_2000 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar10/resnet20/icarl/icarl_2000/'
##;;
##replay)
#python main.py +scenario=task_incremental_cifar10 experiment.experiments=1 +model=resnet20 +training=cifar10 +method=replay_100 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar10/resnet20/replay/replay_100'
#python main.py +scenario=task_incremental_cifar10 experiment.experiments=1 +model=resnet20 +training=cifar10 +method=replay_200 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar10/resnet20/replay/replay_200'
#python main.py +scenario=task_incremental_cifar10 experiment.experiments=1 +model=resnet20 +training=cifar10 +method=replay_500 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar10/resnet20/replay/replay_500'
#python main.py +scenario=task_incremental_cifar10 experiment.experiments=1 +model=resnet20 +training=cifar10 +method=replay_1000 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar10/resnet20/replay/replay_1000'
#;;
#*)
#  echo -n "Unrecognized method"
#esac
