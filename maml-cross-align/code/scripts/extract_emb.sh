# timestamp=$1
epoch=$1

for t in 1 2 3 4 5 6 7
do
	python3 main.py --config-path ../config/s2.json --extract-embeddings --ckpt epoch-$epoch.t$t --task-id $t
done
