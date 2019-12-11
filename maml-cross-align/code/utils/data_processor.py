import numpy as np
import torch
import pickle

# ----------------
import utils.vocab

def makeup_seqs(X, lengths, n):

	ratio = n // X.shape[0]

	if n % X.shape[0]:
		ratio += 1

	X = np.concatenate([X] * ratio, axis=0)
	lengths = np.concatenate([lengths] * ratio, axis=0)
	inds = list(range(X.shape[0]))
	np.random.shuffle(inds)

	sample = np.random.choice(inds, n)

	return X[sample], lengths[sample]


def get_batch_generator(seqs, lengths, batch_size, device, shuffle=True):

	sizes = [seq.shape[0] for seq in seqs]
	n = max(sizes)

	for i in range(len(seqs)):
		if seqs[i].shape[0] < n:
			seqs[i], lengths[i] = makeup_seqs(seqs[i], lengths[i], n)

	num_batches = n // batch_size
	if n % batch_size:
		num_batches += 1

	batch_generator = _batch_generator(seqs, lengths, batch_size, n, num_batches, device, shuffle=True)

	return batch_generator, num_batches


def _batch_generator(seqs, lengths, batch_size, data_size, num_batches, device, shuffle=True):

	b = 0
	inds = list(range(data_size))
	if shuffle:
		np.random.shuffle(inds)
	while True:
		start = b * batch_size
		end = min(data_size, (b + 1) * batch_size)
		seqs_batch = []
		lengths_batch = []
		for seq, length in zip(seqs, lengths):
			seqs_batch.append(seq[inds][start:end])
			lengths_batch.append(length[inds][start:end])

		seqs_batch = np.concatenate(seqs_batch, axis=0)
		lengths_batch = np.concatenate(lengths_batch, axis=0)

		yield torch.tensor(seqs_batch, dtype=torch.int32, device=device), torch.tensor(lengths_batch, dtype=torch.int32, device=device)

		if b == num_batches - 1:
			if shuffle:
				print("shuffling ...")
				np.random.shuffle(inds)
			b = 0
		else:
			b += 1

def get_maml_batch_generator(seqs, lengths, batch_size, num_tasks, device):

	sizes = []
	for seqs_task in seqs:
		sizes += [seq.shape[0] for seq in seqs_task]
	# sizes = [seq.shape[0] for seq in seqs_task for seqs_task in seqs]
	n = max(sizes)

	for t in range(num_tasks):
		seqs_task, lengths_task = seqs[t], lengths[t]
		for i in range(len(seqs_task)):
			if seqs_task[i].shape[0] < n:
				seqs_task[i], lengths_task[i] = makeup_seqs(seqs_task[i], lengths_task[i], n)
		seqs[t], lengths[t] = seqs_task, lengths_task

	num_batches = n // batch_size
	if n % batch_size:
		num_batches += 1

	batch_generator = _maml_batch_generator(seqs, lengths, batch_size, n, num_batches, num_tasks, device)

	return batch_generator, num_batches

def _maml_batch_generator(seqs, lengths, batch_size, data_size, num_batches, num_tasks, device):

	b = 0
	inds = list(range(data_size))
	np.random.shuffle(inds)
	while True:
		seqs_batch_all, lengths_batch_all = [], []
		start = b * batch_size
		end = min(data_size, (b + 1) * batch_size)
		for t in range(num_tasks):
			seqs_batch = []
			lengths_batch = []
			for seq, length in zip(seqs[t], lengths[t]):
				seqs_batch.append(seq[inds][start:end])
				lengths_batch.append(length[inds][start:end])
			seqs_batch = np.concatenate(seqs_batch, axis=0)
			lengths_batch = np.concatenate(lengths_batch, axis=0)

			seqs_batch_all.append(torch.tensor(seqs_batch, dtype=torch.int32, device=device))
			lengths_batch_all.append(torch.tensor(lengths_batch, dtype=torch.int32, device=device))

		yield seqs_batch_all, lengths_batch_all

		if b == num_batches - 1:
			print("shuffling ...")
			np.random.shuffle(inds)
			b = 0
		else:
			b += 1


def get_sequence_lengths(seqs, min_length, max_length):

	tmp = np.concatenate([seqs, np.ones((seqs.shape[0], 1), dtype="int32")], axis=1)
	l = np.clip(np.argmin(tmp, axis=1), min_length, max_length)

	return np.asarray(l, dtype="int32")


# ====================================================================================================
# get data from text files

def get_seq_data_from_file(filename, vocab, mconf):

	with open(filename, 'r', encoding="utf-8") as f:
		lines = f.readlines()

	seqs = np.array(vocab.encode_sents(lines, length=mconf.max_seq_length, pad_token=False), dtype="int32")
	lengths = get_sequence_lengths(seqs, mconf.min_seq_length, mconf.max_seq_length)

	return seqs, np.array(lengths, dtype="int32")


def load_task_data(task_id, data_dir, vocab, label, mconf):

	s0, l0 = get_seq_data_from_file("{}/{}/t{}.0".format(data_dir, label, task_id), vocab, mconf)
	s1, l1 = get_seq_data_from_file("{}/{}/t{}.1".format(data_dir, label, task_id), vocab, mconf)

	return s0, s1, l0, l1


def load_all_tasks_data(mconf, vocab, save=False):

	seqs = {"train": [], "val": []}
	lengths = {"train": [], "val": []}
	for t in range(1, mconf.num_tasks + 1):
		for label in ["train", "val"]:
			print("loading {} data for task {} ...".format(label, t))
			s0, s1, l0, l1 = load_task_data(
				t, mconf.data_dir_prefix, vocab, 
				label=label, mconf=mconf
			)
			seqs[label].append([s0, s1])
			lengths[label].append([l0, l1])

	if save:
		with open(mconf.processed_data_save_dir_prefix + "{}t/vocab".format(mconf.num_tasks), "wb") as f:
			print("saved vocab to {}t/vocab".format(mconf.num_tasks))
			pickle.dump(vocab, f)
		for t in range(mconf.num_tasks):
			data_train, data_val = dict(), dict()
			for s in [0, 1]:
				data_train["s{}".format(s)] = seqs["train"][t][s]
				data_train["l{}".format(s)] = lengths["train"][t][s]
				data_val["s{}".format(s)] = seqs["val"][t][s]
				data_val["l{}".format(s)] = lengths["val"][t][s]

			with open(mconf.processed_data_save_dir_prefix + "{}t/t{}.train".format(mconf.num_tasks, t+1), "wb") as f:
				pickle.dump(data_train, f)
			with open(mconf.processed_data_save_dir_prefix + "{}t/t{}.val".format(mconf.num_tasks, t+1), "wb") as f:
				pickle.dump(data_val, f)

			print("saved data for task {}".format(t+1))

	return seqs, lengths
