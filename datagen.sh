#!/usr/bin/env bash

# 400 moves per round
python main.py play --agents watcher watcher watcher watcher --train 4 --no-gui --n-rounds 100
