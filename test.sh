#!/usr/bin/env bash

usage='
Usage:
	./datagen.sh $MODEL [$SCENARIO]

$MODEL is a vkl model filename.'
[ -z "$1" ] && echo "$usage" && exit
scenario="$2"
[ -z "$2" ] && scenario=classic
export MODEL=$1 # is being read by the script internally
python main.py play --agents vkl --scenario "$scenario" --turn-based
