#!/bin/usr/bash/env bash

for s in 1 2
do
	for t in 1 2 3 4 5 6 7
	do
		for l in 0 1
		do
			echo "s$s t$t l$l"
			kenlm/build/bin/lmplz -o 2 --text s$s/t$t.$l.all > models/s$s/t$t.$l.bigram
		done
	done
done
