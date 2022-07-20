#!/usr/bin/env bash

METHOD=$1
DEVICE=$2

case $METHOD in
gem)
  python main.py +scenario=task_incremental_cifar10 +model=resnet20 +training=cifar10 +method=gem_500 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar10/resnet20/gem/gem_500'
;;
ewc)
  python main.py +scenario=task_incremental_cifar10 +model=resnet20 +training=cifar10 +method=ewc_100 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar10/resnet20/ewc/ewc_100'
;;
oewc)
  python main.py +scenario=task_incremental_cifar10 +model=resnet20 +training=cifar10 +method=oewc_100 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar10/resnet20/oewc/oewc_100'
;;
er)
  python main.py +scenario=task_incremental_cifar10 +model=resnet20 +training=cifar10 +method=er_200 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar10/resnet20/er/er_200'
;;
cml)
  python main.py +scenario=task_incremental_cifar10 +model=resnet20 +training=cifar10 training.epochs=20 +method=cml_01 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar10/resnet20/cml/cml_01'
;;
naive)
  python main.py +scenario=task_incremental_cifar10 +model=resnet20 +training=cifar10  +method=naive optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar10/resnet20/naive/'
;;
cumulative)
  python main.py +scenario=task_incremental_cifar10 +model=resnet20 +training=cifar10 +method=cumulative optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar10/resnet20/cumulative/'
;;
icarl)
  python main.py +scenario=task_incremental_cifar10 +model=resnet20 +training=cifar10 +method=icarl_2000 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar10/resnet20/icarl/icarl_2000/'
;;
replay)
  python main.py +scenario=task_incremental_cifar10 +model=resnet20 +training=cifar10 +method=replay_500 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_cifar10/resnet20/replay/replay_500'
;;
*)
  echo -n "Unrecognized method"
esac
