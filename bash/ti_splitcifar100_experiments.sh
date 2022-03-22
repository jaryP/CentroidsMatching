#!/usr/bin/env bash

METHOD=$1
DEVICE=$2

case $METHOD in
gem)
  python main.py +scenario=task_incremental_cifar100 +model=alexnet +training=cifar100 +method=gem_200 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar100/alexnet/gem/gem_200'
  python main.py +scenario=task_incremental_cifar100 +model=alexnet +training=cifar100 +method=gem_500 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar100/alexnet/gem/gem_500'
  python main.py +scenario=task_incremental_cifar100 +model=alexnet +training=cifar100 +method=gem_1000 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar100/alexnet/gem/gem_1000'
;;
ewc)
  python main.py +scenario=task_incremental_cifar100 +model=alexnet +training=cifar100 +method=ewc_1 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar100/alexnet/ewc/ewc_1'
  python main.py +scenario=task_incremental_cifar100 +model=alexnet +training=cifar100 +method=ewc_10 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar100/alexnet/ewc/ewc_10'
  python main.py +scenario=task_incremental_cifar100 +model=alexnet +training=cifar100 +method=ewc_100 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar100/alexnet/ewc/ewc_100'
  python main.py +scenario=task_incremental_cifar100 +model=alexnet +training=cifar100 +method=ewc_1000 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar100/alexnet/ewc/ewc_1000'
;;
er)
  python main.py +scenario=task_incremental_cifar100 +model=alexnet +training=cifar100 +method=er_200 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar100/alexnet/er/er_200'
  python main.py +scenario=task_incremental_cifar100 +model=alexnet +training=cifar100 +method=er_200_10 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar100/alexnet/er/er_200_10'
  python main.py +scenario=task_incremental_cifar100 +model=alexnet +training=cifar100 +method=er_500_1 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar100/alexnet/er/er_500'
  python main.py +scenario=task_incremental_cifar100 +model=alexnet +training=cifar100 +method=er_500_10 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar100/alexnet/er/er_500_10'
;;
cml)
  python main.py +scenario=task_incremental_cifar100 +model=alexnet +training=cifar100 training.epochs=40 +method=cml_01 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar100/alexnet/cml/cml_01'
  python main.py +scenario=task_incremental_cifar100 +model=alexnet +training=cifar100 training.epochs=40 +method=cml_05 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar100/alexnet/cml/cml_05'
  python main.py +scenario=task_incremental_cifar100 +model=alexnet +training=cifar100 training.epochs=40 +method=cml_1 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar100/alexnet/cml/cml_1'
#  python main.py +scenario=task_incremental_cifar100 +model=alexnet +training=cifar100 training.epochs=40 +method=cml_10 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar100/alexnet/cml/cml_10'
;;
naive)
  python main.py +scenario=task_incremental_cifar100 +model=alexnet +training=cifar100  +method=naive optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar100/alexnet/naive/'
;;
cumulative)
  python main.py +scenario=task_incremental_cifar100 +model=alexnet +training=cifar100 +method=cumulative optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar100/alexnet/cumulative/'
;;
replay)
  python main.py +scenario=task_incremental_cifar100 +model=alexnet +training=cifar100 +method=replay_500 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar100/alexnet/replay/replay_500'
  python main.py +scenario=task_incremental_cifar100 +model=alexnet +training=cifar100 +method=replay_1000 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar100/alexnet/replay/replay_1000'
;;
*)
  echo -n "Unrecognized method"
esac
