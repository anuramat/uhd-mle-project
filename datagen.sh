#!/usr/bin/env bash

# 1000*1024 ~= 4 files, 10G each
# thus 40G total
# which during training takes about 2x, so 80G
# which is close to 100%

usage='
Usage:
	./datagen.sh $MODEL $N_TRANS/1024 [$SCENARIO_NAME]

$MODEL is either "rule_based_agent" or a vkl model filename.
$SCENARIO_NAME is either "coin-heaven" or "classic"/empty.
'
[ -z "$1" ] || [ -z "$2" ] && echo "$usage" && exit

# these are being read in the code (sorry):
export CUDA='yep'
export MODEL=$1
export N_TRANS=$(($2 * 1024))
export SCENARIO_NAME=$3
if [ -z "$SCENARIO_NAME" ]; then
	SCENARIO_NAME="classic"
fi

python main.py play --agents watcher watcher watcher watcher --train 4 --no-gui --n-rounds "$N_TRANS" --scenario "$SCENARIO_NAME"
