#!/bin/usr/bash/env bash

# run multi-bleu, perplexity, and classifier evaluations for a set of model output

model=$1
dir=outputs/$model

echo ">>>>>>> bleu <<<<<<<"

for s in 1 2
do
	for t in 1 2 3 4 5 6 7
	do
		echo "------- test $s task $t -------"
		cat $dir/s$s/t$t.0 $dir/s$s/t$t.1 > $dir/s$s/t$t.val
		cat $dir/s$s/t$t.ref.0 $dir/s$s/t$t.ref.1 > $dir/s$s/t$t.ref
		./multi-bleu.perl $dir/s$s/t$t.ref < $dir/s$s/t$t.val
	done
done

echo ">>>>>>> perplexity <<<<<<<"
cd kenlm
for s in 1 2
do
	for t in 1 2 3 4 5 6 7
	do
		echo "------- test $s task $t -------"
		python3 ./run_eval.py models/s$s/t$t.bigram ../$dir/s$s/t$t.val
	done
done
cd ../

echo ">>>>>>> classifier <<<<<<<"
cd cnn_cls
for s in 1 2
do
	for t in 1 2 3 4 5 6 7
	do
		python3 ./eval.py --eval_train --positive_data_file ../$dir/s$s/t$t.1 --negative_data_file ../$dir/s$s/t$t.0 --checkpoint_dir runs/s$s/t$t/checkpoints/
	done
done
cd ../
