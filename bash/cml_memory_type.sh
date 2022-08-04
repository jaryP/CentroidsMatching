#!/usr/bin/env bash

DEVICE=$1

python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10 +method=cml_01_kmean_2c optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ablation/memory_type/ci_cifar10/resnet20/cml_memory_type/kmeans_2c' experiment.experiments=1
python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10 +method=cml_01_kmean_5c optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ablation/memory_type/ci_cifar10/resnet20/cml_memory_type/kmeans_5c' experiment.experiments=1

python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10 +method=cml_01_spectral_2c optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ablation/memory_type/ci_cifar10/resnet20/cml_memory_type/spectral_2c' experiment.experiments=1
python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10 +method=cml_01_spectral_5c optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ablation/memory_type/ci_cifar10/resnet20/cml_memory_type/spectral_5c' experiment.experiments=1
