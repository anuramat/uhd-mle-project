#!/usr/bin/env bash

# 1000*1024 ~= 4 files, 10G each
# thus 40G total
# which during training takes about 2x, so 80G
# which is close to 100%

usage='
Usage:
	./datagen.sh $MODEL $N_TRANS/1024

$MODEL is either "rule_based_agent" or a vkl model filename.'
[ -z "$1" ] && echo "$usage" && exit
[ -z "$2" ] && echo "$usage" && exit
export MODEL=$1 # is being read by the script internally
export N_TRANS=$(($2 * 1024))
export CUDA='yep'
python main.py play --agents watcher watcher watcher watcher --train 4 --no-gui --n-rounds "$N_TRANS"
