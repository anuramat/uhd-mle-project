#!/usr/bin/env bash

# 100MB of data
# 400 moves * 4 agents * 10 games = 16000 moves
python main.py play --agents watcher watcher watcher watcher --train 4 --no-gui
