#!/usr/bin/env bash

# 400 moves * 4 agents = 1600 moves
python main.py play --agents watcher watcher watcher watcher --train 4 --no-gui --n-rounds 100
