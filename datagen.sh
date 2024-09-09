#!/usr/bin/env bash

# 400 moves per round
export N_GAMES=1000
python main.py play --agents watcher watcher watcher watcher --train 4 --no-gui --n-rounds "$N_GAMES"
