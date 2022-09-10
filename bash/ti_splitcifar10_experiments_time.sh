#!/usr/bin/env bash

METHOD=$1
DEVICE=$2

case $METHOD in
gem)
  python main.py experiment=base1 +scenario=task_incremental_cifar10 +model=resnet20 +training=cifar10 training.epochs=3 +method=gem_500 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/time/ci_cifar10/resnet20/gem/gem_500'
;;
ewc)
  python main.py experiment=base1 +scenario=task_incremental_cifar10 +model=resnet20 +training=cifar10 training.epochs=3 +method=ewc_100 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/time/ci_cifar10/resnet20/ewc/ewc_100'
;;
oewc)
  python main.py experiment=base1 +scenario=task_incremental_cifar10 +model=resnet20 +training=cifar10 training.epochs=3 +method=oewc_100 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/time/ci_cifar10/resnet20/oewc/oewc_100'
;;
cml)
  python main.py experiment=base1 +scenario=task_incremental_cifar10 +model=resnet20 +training=cifar10 training.epochs=3 +method=cml_01 optimizer=sgd +method.dev_split_size=50 training.device="$DEVICE" hydra.run.dir='./results/time/ci_cifar10/resnet20/cml/cml_01'
;;
naive)
  python main.py experiment=base1 +scenario=task_incremental_cifar10 +model=resnet20 +training=cifar10 training.epochs=3  +method=naive optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/time/ci_cifar10/resnet20/naive/'
;;
cumulative)
  python main.py experiment=base1 +scenario=task_incremental_cifar10 +model=resnet20 +training=cifar10 training.epochs=3 +method=cumulative optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/time/ci_cifar10/resnet20/cumulative/'
;;
replay)
  python main.py experiment=base1 +scenario=task_incremental_cifar10 +model=resnet20 +training=cifar10 training.epochs=3 +method=replay_500 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/time/ci_cifar10/resnet20/replay/replay_500'
;;
er)
  python main.py experiment=base1 +scenario=task_incremental_cifar10 +model=resnet20 +training=cifar10 training.epochs=3 +method=er_200 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/time/ti_cifar10/resnet20/er/er_200'
;;
*)
  echo -n "Unrecognized method"
esac
