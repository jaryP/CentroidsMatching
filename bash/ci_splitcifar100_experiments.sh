#!/usr/bin/env bash

METHOD=$1
DEVICE=$2

case $METHOD in
gem)
  python main.py +scenario=class_incremental_cifar100 +model=resnet20 +training=cifar100 +method=gem_1000 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar100/resnet20/gem/gem_1000'
;;
ewc)
  python main.py +scenario=class_incremental_cifar100 +model=resnet20 +training=cifar100 +method=ewc_100 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar100/resnet20/ewc/ewc_100'
;;
oewc)
  python main.py +scenario=class_incremental_cifar100 +model=resnet20 +training=cifar100 +method=oewc_100 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar100/resnet20/oewc/oewc_100'
;;
cml)
  python main.py +scenario=class_incremental_cifar100 +model=resnet20 +training=cifar100 +method=cml_01 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar100/resnet20/cml_ablation/cml_01_proj1'
;;
naive)
  python main.py +scenario=class_incremental_cifar100 +model=resnet20 +training=cifar100  +method=naive optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar100/resnet20/naive/'
;;
cumulative)
  python main.py +scenario=class_incremental_cifar100 +model=resnet20 +training=cifar100 +method=cumulative optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar100/resnet20/cumulative/'
;;
icarl)
  python main.py +scenario=class_incremental_cifar100 +model=resnet20 +training=cifar100 +method=icarl_2000 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar100/resnet20/icarl/icarl_2000/'
;;
replay)
  python main.py +scenario=class_incremental_cifar100 +model=resnet20 +training=cifar100 +method=replay_500 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar100/resnet20/replay/replay_500'
;;
ssil)
  python main.py experiment=base1 experiment.load=false training.epochs=100 +scenario=class_incremental_cifar100 +model=resnet20 +training=cifar10 +method=ssil optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar100/resnet20/ssil/ssil_500'
;;
cope)
  python main.py +scenario=class_incremental_cifar100 +model=resnet20 +training=cifar10 +method=cope optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar100/resnet20/cope/cope_2000'
;;
*)
  echo -n "Unrecognized method"
esac
