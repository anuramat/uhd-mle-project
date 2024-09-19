#!/usr/bin/env bash

usage='usage: ./datagen.sh $N_TRANS $MODEL <$MODEL_FILE>'
[ -z "$1" ] && echo "$usage" && exit
[ -z "$2" ] && echo "$usage" && exit
export N_TRANS=$1
export MODEL=$2 # is being read by the script internally
export MODEL_FILE=$3
python main.py play --agents watcher watcher watcher watcher --train 4 --no-gui --n-rounds "$N_TRANS"
