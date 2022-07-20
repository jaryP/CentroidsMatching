#!/usr/bin/env bash

METHOD=$1
DEVICE=$2

case $METHOD in
gem)
  python main.py experiment=base2 +scenario=task_incremental_tinyimagenet +model=resnet20 +training=tinyimagenet +method=gem_1000 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_tinyimagenet/resnet20/gem/gem_1000'
;;
ewc)
  python main.py experiment=base2 +scenario=task_incremental_tinyimagenet +model=resnet20 +training=tinyimagenet +method=ewc_100 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_tinyimagenet/resnet20/ewc/ewc_100'
;;
oewc)
  python main.py experiment=base2 +scenario=task_incremental_tinyimagenet +model=resnet20 +training=tinyimagenet +method=oewc_100 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_tinyimagenet/resnet20/oewc/oewc_100'
;;
er)
  python main.py experiment=base2 +scenario=task_incremental_tinyimagenet +model=resnet20 +training=tinyimagenet +method=er_200 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_tinyimagenet/resnet20/er/er_200'
;;
cml)
  python main.py experiment=base2 +scenario=task_incremental_tinyimagenet +model=resnet20 +training=tinyimagenet +method=cml_01 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_tinyimagenet/resnet20/cml/cml_01' experiment.load=fasle
;;
naive)
  python main.py experiment=base2 +scenario=task_incremental_tinyimagenet +model=resnet20 +training=tinyimagenet  +method=naive optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_tinyimagenet/resnet20/naive/'
;;
cumulative)
  python main.py experiment=base2 +scenario=task_incremental_tinyimagenet +model=resnet20 +training=tinyimagenet +method=cumulative optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_tinyimagenet/resnet20/cumulative/'
;;
replay)
  python main.py experiment=base2 +scenario=task_incremental_tinyimagenet +model=resnet20 +training=tinyimagenet +method=replay_500 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_tinyimagenet/resnet20/replay/replay_500'
;;
*)
  echo -n "Unrecognized method"
esac
