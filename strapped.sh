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
	*) ;;
esac

in=in.pt
out=out.pt
model="$in"
[ "$1" = 0 ] && model=none

while true; do
	[ "$i" != 0 ] && model=source_model.pt
	echo "~~~~~~~~~~~~~~~~~ Generation $i ~~~~~~~~~~~~~~~~~~~~~~"
	# generate data
	first_iteration="false"
	[ "$1" = "$i" ] && first_iteration="true"
	[ "$first_iteration" = "false" ] || [ "$skip_first_datagen" = "false" ] && {
		[ "$model" != "none" ] && ./datagen.sh source_model.pt 400
		./datagen.sh rule_based_agent 200
		./datagen.sh rule_based_agent 200 coin-heaven
	}
	# start training (start from scratch on zeroth gen)
	./train.py --n-epochs 4 --input "$model" --output "$out"
	model="$in"
	# backup the model
	cp "$out" "./output/gen${i}.pt"
	mv "$out" "$in"
	echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
	((i++))
done
