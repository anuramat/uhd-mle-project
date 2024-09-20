#!/usr/bin/env bash

usage='
Usage:
	./datagen.sh $MODEL

$MODEL is a vkl model filename.'
[ -z "$1" ] && echo "$usage" && exit
export MODEL=$1 # is being read by the script internally
# export CUDA='yep'
python main.py play --agents vkl vkl vkl vkl
