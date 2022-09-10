#!/usr/bin/env bash

METHOD=$1
DEVICE=$2

case $METHOD in
gem)
  python main.py experiment=base2 +scenario=class_incremental_tinyimagenet +model=resnet20 +training=tinyimagenet +method=gem_1000 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_tinyimagenet/resnet20/gem/gem_1000'
;;
ewc)
  python main.py experiment=base2 +scenario=class_incremental_tinyimagenet +model=resnet20 +training=tinyimagenet +method=ewc_100 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_tinyimagenet/resnet20/ewc/ewc_100'
;;
oewc)
  python main.py experiment=base2 +scenario=class_incremental_tinyimagenet +model=resnet20 +training=tinyimagenet +method=oewc_100 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_tinyimagenet/resnet20/oewc/oewc_100'
;;
cml)
 python main.py experiment=base2 +scenario=class_incremental_tinyimagenet +model=resnet20 +training=tinyimagenet +method=cml_01 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_tinyimagenet/resnet20/cml/cml_01'
;;
naive)
  python main.py experiment=base2 +scenario=class_incremental_tinyimagenet +model=resnet20 +training=tinyimagenet  +method=naive optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_tinyimagenet/resnet20/naive/'
;;
cumulative)
  python main.py experiment=base2 +scenario=class_incremental_tinyimagenet +model=resnet20 +training=tinyimagenet +method=cumulative optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_tinyimagenet/resnet20/cumulative/'
;;
replay)
  python main.py experiment=base2 +scenario=class_incremental_tinyimagenet +model=resnet20 +training=tinyimagenet +method=replay_1000 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_tinyimagenet/resnet20/replay/replay_1000'
;;
ssil)
  python main.py experiment=base1 experiment.load=false training.epochs=100 experiment=base2 +scenario=class_incremental_tinyimagenet +model=resnet20 +training=tinyimagenet +method=ssil method.memory_size=1000 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_tinyimagenet/resnet20/ssil/ssil_1000'
;;
cope)
  python main.py experiment=base2 +scenario=class_incremental_tinyimagenet +model=resnet20 +training=tinyimagenet +method=cope optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_tinyimagenet/resnet20/cope/cope_500'
;;
*)
  echo -n "Unrecognized method"
esac
