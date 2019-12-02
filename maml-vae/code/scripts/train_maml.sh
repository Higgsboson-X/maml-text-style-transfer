#!/bin/usr/bash/env bash
# translations
# python3 main.py --corpus translations --maml-epochs 14 --transfer-epochs 6 --epochs-per-val 2 --config-path ../config/translations.json --maml-batch-size 8 --sub-batch-size 32 --train-batch-size 32 --inference

for corpus in yelp amazon
do
	echo ">>>>>>> $corpus <<<<<<<"
	python3 main.py --corpus $corpus --maml-epochs 14 --transfer-epochs 6 --epochs-per-val 2 --config-path ../config/${corpus}.json --maml-batch-size 16 --sub-batch-size 64 --train-batch-size 64
done
