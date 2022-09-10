#!/usr/bin/env bash

DEVICE=$1

declare -a arr=("gem" "ewc" "cml" "naive" "cumulative" "replay" "oewc" "ssil" "cope" "er")

for i in "${arr[@]}"
do
  bash bash/ti_splitcifar10_experiments_time.sh "$i" "$DEVICE"
done

for i in "${arr[@]}"
do
  bash bash/ci_splitcifar10_experiments_time.sh "$i" "$DEVICE"
done
