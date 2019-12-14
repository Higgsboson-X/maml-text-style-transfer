model=$1
sample_size=$2
# only for cross-align(pretrain) + vae(pretrain) + maml-{vae, cross-align}
for t in 1 2 3 4 5 6 7
do
	embedding_path=outputs/$model/s2.emb/t$t.emb
	output_path=outputs/$model/s2.emb/
	index=t$t
	python3 tsne/tsne.py $embedding_path $output_path $index $sample_size
done
