#!/usr/bin/env bash

# 1*1024 ~= 34M over 4 files
# 200*1024 ~= 7.1G over 4 files

usage='usage: ./datagen.sh $N_TRANS $MODEL <$MODEL_FILE>'
[ -z "$1" ] && echo "$usage" && exit
[ -z "$2" ] && echo "$usage" && exit
export N_TRANS=$(expr "$1" \* 1024)
export MODEL=$2 # is being read by the script internally
export MODEL_FILE=$3
python main.py play --agents watcher watcher watcher watcher --train 4 --no-gui --n-rounds "$N_TRANS"
