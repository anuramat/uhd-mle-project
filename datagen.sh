#!/usr/bin/env bash

# 1*1024 ~= 34M over 4 files
# 200*1024 ~= 7.1G over 4 files

usage='
Usage:
	./datagen.sh $MODEL $N_TRANS/1024

$MODEL is either "rule_based_agent" or a vkl model filename.'
[ -z "$1" ] && echo "$usage" && exit
[ -z "$2" ] && echo "$usage" && exit
export MODEL=$1 # is being read by the script internally
export N_TRANS=$(($2 * 1024))

python main.py play --agents watcher watcher watcher watcher --train 4 --no-gui --n-rounds "$N_TRANS"
