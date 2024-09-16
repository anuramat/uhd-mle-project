#!/usr/bin/env bash

usage='usage: ./datagen.sh $MODEL $N_GAMES'
[ -z "$1" ] && echo "$usage" && exit
[ -z "$2" ] && echo "$usage" && exit
export MODEL=$1 # is being read by the script internally
export N_GAMES=$2
python main.py play --agents watcher watcher watcher watcher --train 4 --no-gui --n-rounds "$N_GAMES"
