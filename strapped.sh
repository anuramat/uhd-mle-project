#!/usr/bin/env bash

# # set -e
# I'd have to verify that training/data generation both return 0
# also first rm usually fails

usage='
Usage:
	./strapped.sh $START_GEN'
[ -z "$1" ] && echo "$usage" && exit
i="$1"

read -rp "skip first datagen (y/n)? " choice
case "$choice" in
	y) skip_first_datagen="true" ;;
	n) skip_first_datagen="false" ;;
	*)
		echo "huh"
		exit
		;;
esac

while true; do
	echo "~~~~~~~~~~~~~~~~~ Generation $i ~~~~~~~~~~~~~~~~~~~~~~"
	# generate data
	[ "$1" != "$i" ] || [ "$skip_first_datagen" = "false" ] && {
		./datagen.sh source_model.pt 400
		./datagen.sh rule_based_agent 200
		./datagen.sh rule_based_agent 200 coin-heaven
	}
	# start training
	./train.py --n-epochs 4
	# backup the model
	cp ./result_model.pt "./output/gen${i}.pt"
	mv ./result_model.pt ./source_model.pt
	echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
	((i++))
done
