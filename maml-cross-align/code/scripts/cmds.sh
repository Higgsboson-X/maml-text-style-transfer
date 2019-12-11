s=$1
t=$2

# make text.pretrain, t1.all, t2.all, etc.
bash scripts/get_pretrain_text.sh s$s

# make dirs (with 7 tasks)
bash scripts/make_dirs.sh s$s 7

# --------------------------------
# run original cross_align

# :initial
python3 original.py --config-path ../config/s$s.json --corpus s$s --task-id $t --batch-size 64 --epochs 20 --epochs-per-val 5 --pretrain-epochs 10
# :load last ckpt / load processed data (from saved config)
python3 original.py --config-path ../output/s$s/t$t.json --corpus s$s --task-id $t --batch-size 64 --epochs 20 --epochs-per-val 5 --pretrain-epochs 5 --load-model --load-data

# no inference for original model

# --------------------------------
# run maml_cross_align
# :initial
python3 main.py --config-path ../config/s$s.json  --corpus s$s --maml-epochs 20 --transfer-epochs 10 --epochs-per-val 5 maml-batch-size 32 --sub-batch-size 64 --train-batch-size 64
# :load last ckpt / load processed data
python3 main.py --config-path ../config/s$s.json --corpus s$s --maml-epochs  20 --transfer-epochs 10 --epochs-per-val 5 --maml-batch-size 32 --sub-batch-size 64 --train-batch-size 64

# run inference
python3 main.py --config-path ../config/s$s.json --corpus s$s --task-id $t

# extract embeddings
python3 main.py --config-path ../config/s$s.json --corpus s$s --extract-embeddings --task-id $t --ckpt epoch-1.t$t # if from pretrain to load vocab: --from-pretrain

# online-inference
python3 main.py --online-inference --config-path ../config/s$s.json --corpus s$s --ckpt epoch-1.t$t

# DONE!

