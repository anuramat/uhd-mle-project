#!/usr/bin/env bash

plain=100
coin=20
echo "Generating $plain (1/2)"
./datagen.sh rule_based_agent "$plain"
echo "Generating $coin (2/2)"
./datagen.sh rule_based_agent "$coin" coin-heaven
./train.py --n-epochs 40 --input none --output "in.pt" --lr 3e-4
cp in.pt ./output/gen0.pt
