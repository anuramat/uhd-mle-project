#!/usr/bin/env bash

# 400 moves per round
[ -z "$1" ] && echo "provide number of rounds" && exit
export N_GAMES=$1
export MODEL="rule_based_agent"
python main.py play --agents watcher watcher watcher watcher --train 4 --no-gui --n-rounds "$N_GAMES"
