#!/usr/bin/env bash

METHOD=$1
DEVICE=$2

case $METHOD in
gem)
#  python main.py +scenario=class_incremental_cifar100 +model=resnet20 +training=cifar100 +method=gem_200 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar100/resnet20/gem/gem_200'
  python main.py +scenario=class_incremental_cifar100 +model=resnet20 +training=cifar100 +method=gem_500 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar100/resnet20/gem/gem_500'
#  python main.py +scenario=class_incremental_cifar100 +model=resnet20 +training=cifar100 +method=gem_1000 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar100/resnet20/gem/gem_1000'
;;
ewc)
#  python main.py +scenario=class_incremental_cifar100 +model=resnet20 +training=cifar100 +method=ewc_1 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar100/resnet20/ewc/ewc_1'
#  python main.py +scenario=class_incremental_cifar100 +model=resnet20 +training=cifar100 +method=ewc_10 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar100/resnet20/ewc/ewc_10'
  python main.py +scenario=class_incremental_cifar100 +model=resnet20 +training=cifar100 +method=ewc_100 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar100/resnet20/ewc/ewc_100'
#  python main.py +scenario=class_incremental_cifar100 +model=resnet20 +training=cifar100 +method=ewc_1000 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar100/resnet20/ewc/ewc_1000'
;;
oewc)
#  python main.py +scenario=class_incremental_cifar100 +model=resnet20 +training=cifar100 +method=ewc_1 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar100/resnet20/ewc/ewc_1'
#  python main.py +scenario=class_incremental_cifar100 +model=resnet20 +training=cifar100 +method=ewc_10 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar100/resnet20/ewc/ewc_10'
  python main.py +scenario=class_incremental_cifar100 +model=resnet20 +training=cifar100 +method=oewc_100 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar100/resnet20/oewc/oewc_100'
#  python main.py +scenario=class_incremental_cifar100 +model=resnet20 +training=cifar100 +method=ewc_1000 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar100/resnet20/ewc/ewc_1000'
;;
er)
#  python main.py +scenario=class_incremental_cifar100 +model=resnet20 +training=cifar100 +method=er_200 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar100/resnet20/er/er_200'
#  python main.py +scenario=class_incremental_cifar100 +model=resnet20 +training=cifar100 +method=er_200_10 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar100/resnet20/er/er_200_10'
#  python main.py +scenario=class_incremental_cifar100 +model=resnet20 +training=cifar100 +method=er_500_1 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar100/resnet20/er/er_500'
#  python main.py +scenario=class_incremental_cifar100 +model=resnet20 +training=cifar100 +method=er_500_10 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar100/resnet20/er/er_500_10'
;;
cml)
#  python main.py +scenario=class_incremental_cifar100 +model=resnet20 +training=cifar100  +method=cml_1 +method.proj_w=10 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar100/resnet20/cml/cml_1_proj10' experiment.experiments=1
#  python main.py +scenario=class_incremental_cifar100 +model=resnet20 +training=cifar100 +method=cml_01 +method.proj_w=1000 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar100/resnet20/cml_ablation/cml_01_proj1000' experiment.experiments=1
#  python main.py +scenario=class_incremental_cifar100 +model=resnet20 +training=cifar100 +method=cml_01 +method.proj_w=100 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar100/resnet20/cml_ablation/cml_01_proj100' experiment.experiments=1
  python main.py +scenario=class_incremental_cifar100 +model=resnet20 +training=cifar100 +method=cml_01 +method.proj_w=1 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar100/resnet20/cml_ablation/cml_01_proj1'
#  python main.py +scenario=class_incremental_cifar100 +model=resnet20 +training=cifar100 training.epochs=40 +method=cml_1 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar100/resnet20/cml/cml_1'
#  python main.py +scenario=class_incremental_cifar100 +model=resnet20 +training=cifar100 +method.proj_w=10 training.epochs=10 +method=cml_01 optimizer=sgd  +method.sit_memory_size=1000 training.device="$DEVICE" hydra.run.dir='./results/ci_cifar100/resnet20/cml/cml_10' experiment.experiments=1
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
#  python main.py +scenario=class_incremental_cifar100 +model=resnet20 +training=cifar100 +method=replay_1000 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar100/resnet20/replay/replay_1000'
;;
*)
  echo -n "Unrecognized method"
esac
