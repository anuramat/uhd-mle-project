#!/usr/bin/env bash

source_episodes=200 # about 8G
epochs=64
for i in {4..100}; do
	echo "~~~~~~~~~~~~~~~~~ Generation $i {{{1 ~~~~~~~~~~~~~~~~~"
	# generate data
	rm agent_code/watcher/data/source*.pt
	./datagen.sh source_model.pt "$source_episodes"
	# start training
	./train.py --n-epochs "$epochs"
	# backup the model
	cp ./result_model.pt "$HOME/weights/gen${i}.pt"
	mv ./result_model.pt ./source_model.pt
	printf '\n\n\n'
done
