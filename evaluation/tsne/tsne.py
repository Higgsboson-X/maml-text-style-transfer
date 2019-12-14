import pickle
import sys
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE

def eval_tsne(embeddings, output_dir_prefix, index, sample_size):

	print("sampling {} embeddings ...".format(sample_size))

	inds0 = list(range(embeddings[0].shape[0]))
	inds1 = list(range(embeddings[1].shape[0]))

	inds0 = np.random.choice(inds0, sample_size, replace=False)
	inds1 = np.random.choice(inds1, sample_size, replace=False)

	sampled_embeddings = np.concatenate([embeddings[0][inds0], embeddings[1][inds1]], axis=0)
	inds1 += inds0.shape[0]

	print("sampled embeddings: ", inds0.shape[0], inds1.shape[0], "total: ", sampled_embeddings.shape)
	
	coords = TSNE(n_components=2).fit_transform(X=sampled_embeddings)

	inds = list(range(inds0.shape[0] + inds1.shape[0]))

	plot_tsne_coords(coords, output_path, [inds[:inds0.shape[0]], inds[inds0.shape[0]:]], index)

	with open(output_path + index + ".coord", "wb") as f:
		pickle.dump((coords, inds0, inds1), f)

	print("dumped TSNE coordinates")


def plot_tsne_coords(coords, output_path, inds_list, index):

	colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
	markers = ['x', '+']

	matplotlib.use("svg")

	plt.figure(index)
	for i in range(len(inds_list)):
		plt.scatter(
			x=coords[np.asarray(inds_list[i]), 0],
			y=coords[np.asarray(inds_list[i]), 1],
			marker=markers[i % len(markers)],
			c=colors[i % len(colors)],
			label=str(i),
			alpha=0.75
		)

	plt.legend(loc="upper right", fontsize="x-large")
	plt.axis("off")
	fig_path = output_path + index + ".tsne"
	plt.savefig(fname=fig_path, format="svg", bbox_inches="tight", transparent=False)
	plt.close()



if __name__ == "__main__":

	embedding_path = sys.argv[1]
	output_path = sys.argv[2]
	index = sys.argv[3]
	sample_size = int(sys.argv[4])

	with open(embedding_path, "rb") as f:
		embeddings = pickle.load(f)

	style_embeddings, content_embeddings = embeddings["style"], embeddings["content"]

	eval_tsne(style_embeddings, output_path, index + ".s", min(sample_size, style_embeddings[0].shape[0], style_embeddings[1].shape[0]))
	eval_tsne(content_embeddings, output_path, index + ".c", min(sample_size, content_embeddings[0].shape[0], content_embeddings[1].shape[0]))

