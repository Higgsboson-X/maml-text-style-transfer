#!/bin/usr/bash/env bash
for s in 1 2
do
	for t in 1 2 3 4 5 6 7
	do
		echo ">>>>>>> s$s t$t <<<<<<<"
		python3 train.py --positive_data_file data/s$s/t$t.1 --negative_data_file data/s$s/t$t.0 --num_epochs 7 --corpus_name s$s/t$t --batch_size 128
	done
done
