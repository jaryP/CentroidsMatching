#!/usr/bin/env bash

METHOD=$1
DEVICE=$2

case $METHOD in
gem)
#  python main.py +scenario=task_incremental_tinyimagenet +model=resnet20 +training=tinyimagenet +method=gem_200 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_tinyimagenet/resnet20/gem/gem_200'
  python main.py +scenario=task_incremental_tinyimagenet +model=resnet20 +training=tinyimagenet +method=gem_500 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_tinyimagenet/resnet20/gem/gem_500'
#  python main.py +scenario=task_incremental_tinyimagenet +model=resnet20 +training=tinyimagenet +method=gem_1000 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_tinyimagenet/resnet20/gem/gem_1000'
;;
ewc)
#  python main.py +scenario=task_incremental_tinyimagenet +model=resnet20 +training=tinyimagenet +method=ewc_1 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_tinyimagenet/resnet20/ewc/ewc_1'
#  python main.py +scenario=task_incremental_tinyimagenet +model=resnet20 +training=tinyimagenet +method=ewc_10 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_tinyimagenet/resnet20/ewc/ewc_10'
  python main.py +scenario=task_incremental_tinyimagenet +model=resnet20 +training=tinyimagenet +method=ewc_100 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_tinyimagenet/resnet20/ewc/ewc_100'
  python main.py +scenario=task_incremental_tinyimagenet +model=resnet20 +training=tinyimagenet +method=ewc_1000 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_tinyimagenet/resnet20/ewc/ewc_1000'
;;
oewc)
#  python main.py +scenario=task_incremental_tinyimagenet +model=resnet20 +training=tinyimagenet +method=ewc_1 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_tinyimagenet/resnet20/ewc/ewc_1'
#  python main.py +scenario=task_incremental_tinyimagenet +model=resnet20 +training=tinyimagenet +method=ewc_10 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_tinyimagenet/resnet20/ewc/ewc_10'
  python main.py +scenario=task_incremental_tinyimagenet +model=resnet20 +training=tinyimagenet +method=oewc_100 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_tinyimagenet/resnet20/oewc/oewc_100'
#  python main.py +scenario=task_incremental_tinyimagenet +model=resnet20 +training=tinyimagenet +method=ewc_1000 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_tinyimagenet/resnet20/ewc/ewc_1000'
;;
er)
#  python main.py +scenario=task_incremental_tinyimagenet +model=resnet20 +training=tinyimagenet +method=er_200 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_tinyimagenet/resnet20/er/er_200'
#  python main.py +scenario=task_incremental_tinyimagenet +model=resnet20 +training=tinyimagenet +method=er_200_10 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_tinyimagenet/resnet20/er/er_200_10'
#  python main.py +scenario=task_incremental_tinyimagenet +model=resnet20 +training=tinyimagenet +method=er_500_1 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_tinyimagenet/resnet20/er/er_500'
#  python main.py +scenario=task_incremental_tinyimagenet +model=resnet20 +training=tinyimagenet +method=er_500_10 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_tinyimagenet/resnet20/er/er_500_10'
;;
cml)
#  python main.py +scenario=task_incremental_tinyimagenet +model=resnet20 +training=tinyimagenet training.epochs=20 +method=cml_05 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_tinyimagenet/resnet20/cml/cml_05'
  python main.py +scenario=task_incremental_tinyimagenet +model=resnet20 +training=tinyimagenet +method=cml_1 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_tinyimagenet/resnet20/cml/cml_1'
  python main.py +scenario=task_incremental_tinyimagenet +model=resnet20 +training=tinyimagenet +method=cml_01 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_tinyimagenet/resnet20/cml/cml_01'
  python main.py +scenario=task_incremental_tinyimagenet +model=resnet20 +training=tinyimagenet +method=cml_10 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_tinyimagenet/resnet20/cml/cml_10'
;;
naive)
  python main.py +scenario=task_incremental_tinyimagenet +model=resnet20 +training=tinyimagenet  +method=naive optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_tinyimagenet/resnet20/naive/'
;;
cumulative)
  python main.py +scenario=task_incremental_tinyimagenet +model=resnet20 +training=tinyimagenet +method=cumulative optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_tinyimagenet/resnet20/cumulative/'
;;
replay)
  python main.py +scenario=task_incremental_tinyimagenet +model=resnet20 +training=tinyimagenet +method=replay_500 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_tinyimagenet/resnet20/replay/replay_500'
#  python main.py +scenario=task_incremental_tinyimagenet +model=resnet20 +training=tinyimagenet +method=replay_1000 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ti_tinyimagenet/resnet20/replay/replay_1000'
;;
*)
  echo -n "Unrecognized method"
esac
