import pickle
import os
import numpy as np

# ----------------
import utils.data_processor
import utils.vocab

def print_data_info(vocab, seqs, lengths, num_tasks):

	print("vocab_size = {}".format(vocab._size))
	for key in ["train", "val"]:
		print("{}\n-------".format(key))
		for t in range(num_tasks):
			print("task {} data:".format(t+1))
			for s in [0, 1]:
				print("\t {}:".format(s), seqs[key][t][s].shape, lengths[key][t][s].shape)


def load_data(mconf, load_data=False, save=False):
	
	if load_data:
		with open(mconf.processed_data_save_dir_prefix + "{}t/vocab".format(mconf.num_tasks), "rb") as f:
			vocab = pickle.load(f)
		seqs = {"train": [], "val": []}
		lengths = {"train": [], "val": []}
		for t in range(mconf.num_tasks):
			for label in ["train", "val"]:
				with open(mconf.processed_data_save_dir_prefix + "{}t/t{}.{}".format(mconf.num_tasks, t+1, label), "rb") as f:
					data = pickle.load(f)
					s0, s1 = data["s0"], data["s1"]
					l0, l1 = data["l0"], data["l1"]
					seqs[label].append([s0, s1])
					lengths[label].append([l0, l1])
	else:
		vocab, seqs, lengths = _load_data(mconf, save=save)

	mconf.vocab_size = vocab._size

	return vocab, seqs, lengths


def _load_data(mconf, save=False):

	vocab = utils.vocab.Vocabulary(mconf=mconf)
	if os.path.exists(mconf.data_dir_prefix + "text.pretrain"):
		print("updating vocab from {} ...".format(mconf.data_dir_prefix + "text.pretrain"))
		vocab.update_vocab(mconf.data_dir_prefix + "text.pretrain")
	else:
		for t in range(mconf.num_tasks):
			print("updating vocab from task {} ...".format(t+1))
			for s in [0, 1]:
				vocab.update_vocab(mconf.data_dir_prefix + "train/t{}.{}".format(t+1, s))
				vocab.update_vocab(mconf.data_dir_prefix + "val/t{}.{}".format(t+1, s))

	seqs, lengths = utils.data_processor.load_all_tasks_data(mconf, vocab, save)

	return vocab, seqs, lengths


def load_embedding_from_wdv(vocab, path):

	emb_size = None
	with open(path, 'r', encoding="utf-8") as f:
		for line in f:
			line = line.split()
			word = line[0]
			emb = np.array(line[1:], dtype="float32")
			if emb_size is None:
				emb_size = emb.shape[0]
				embedding = np.asarray(np.random.normal(size=(vocab._size, emb_size)), dtype="float32")
			if word in vocab._word2id:
				embedding[vocab._word2id[word]] = emb

	return embedding
