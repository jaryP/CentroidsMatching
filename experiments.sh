#!/usr/bin/env bash

SCENARIO=$1
METHOD=$2
DEVICE=$3

case $SCENARIO in
  splitcifar10)
    case $METHOD in
    gem)
      python main.py +scenario=task_incremental_cifar10 +model=alexnet +training=cifar10 +method=gem_200 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/splitcifar10/gem/gem_200'
      python main.py +scenario=task_incremental_cifar10 +model=alexnet +training=cifar10 +method=gem_500 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/splitcifar10/gem/gem_500'
      python main.py +scenario=task_incremental_cifar10 +model=alexnet +training=cifar10 +method=gem_1000 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/splitcifar10/gem/gem_1000'
    ;;
    cml)
      python main.py +scenario=task_incremental_cifar10 +model=alexnet +training=cifar10 +method=cml_1 optimizer=adam  training.device="$DEVICE" hydra.run.dir='./results/splitcifar10/cml/cml_1'
      python main.py +scenario=task_incremental_cifar10 +model=alexnet +training=cifar10 +method=cml_10 optimizer=adam  training.device="$DEVICE" hydra.run.dir='./results/splitcifar10/cml/cml_10'
    ;;
    *)
      echo -n "Unrecognized method"
    esac
  ;;
  *)
  echo -n "Unrecognized scenario"
esac
