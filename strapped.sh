#!/usr/bin/env bash

# # set -e
# I'd have to verify that training/data generation both return 0
# also first rm usually fails

usage='
Usage:
	./strapped.sh $START_GEN'
[ -z "$1" ] && echo "$usage" && exit
source_episodes=200 # about 8G
epochs=64
i="$1"
while true; do
	echo "~~~~~~~~~~~~~~~~~ Generation $i ~~~~~~~~~~~~~~~~~~~~~~"
	# generate data
	rm agent_code/watcher/data/source*.pt
	./datagen.sh source_model.pt "$source_episodes"
	# start training
	./train.py --n-epochs "$epochs"
	# backup the model
	cp ./result_model.pt "./output/gen${i}.pt"
	mv ./result_model.pt ./source_model.pt
	echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
	((i++))
done
