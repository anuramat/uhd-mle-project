#!/usr/bin/env bash

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

# TODO separate bootstrap script from dqn (maybe)

while true; do
	echo "~~~~~~~~~~~~~~~~~ Generation $i ~~~~~~~~~~~~~~~~~~~~~~"

	# generate data
	# unless we already did
	first_iteration="false"
	[ "$1" = "$i" ] && first_iteration="true"
	[ "$first_iteration" = "false" ] || [ "$skip_first_datagen" = "false" ] && {

		./datagen.sh "$in" 100            # 7 minutes
		./datagen.sh "$in" 20 coin-heaven # 2 min
		SHADOW=1 ./datagen.sh "$in" 100

	}

	./train.py --n-epochs 128 --input "$in" --output "$out" --lr 3e-5

	# backup the model
	cp "$out" "./output/gen${i}.pt"
	mv "$out" "$in"

	echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
	((i++))
done
