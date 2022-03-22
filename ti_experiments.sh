#!/usr/bin/env bash

DEVICE=$1

declare -a arr=("gem" "ewc" "er" "cml" "naive" "cumulative" "replay" "joint")

for i in "${arr[@]}"
do
  bash bash/ti_splitcifar10_experiments.sh "$i" "$DEVICE"
done

for i in "${arr[@]}"
do
  bash bash/ti_splitcifar100_experiments.sh "$i" "$DEVICE"
done

for i in "${arr[@]}"
do
  bash bash/ti_splittinyimagenet_experiments.sh "$i" "$DEVICE"
done
