#!/usr/bin/env bash

DEVICE=$1

declare -a arr=("gem" "ewc" "er" "cml" "naive" "cumulative" "replay")

for i in "${arr[@]}"
do
  bash bash/ci_splitcifar10_experiments.sh "$i" "$DEVICE"
done

for i in "${arr[@]}"
do
  bash bash/ci_splitcifar100_experiments.sh "$i" "$DEVICE"
done

#for i in "${arr[@]}"
#do
#  bash bash/ci_splittinyimagenet_experiments.sh "$i" "$DEVICE"
#done
