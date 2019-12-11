import sys
import json
import pprint
import pickle
import numpy as np
import datetime as dt

# ----------------
import config.model_config
import models.cross_align
import models.maml_cross_align
import utils.data_loader
import utils.data_processor

def _train_maml(net, mconf, seqs, lengths, vocab, total_epochs=10, epochs_per_val=2, support_batch_size=32, query_batch_size=8):

	print("maml learning ...")
	turns = total_epochs // epochs_per_val
	if total_epochs % epochs_per_val:
		turns += 1

	for turn in range(turns):
		init_epoch = turn * epochs_per_val
		end_epoch = min(total_epochs, (turn + 1) * epochs_per_val)
		net.train_maml(
			support_input_sequence=seqs["train"], 
			support_lengths=lengths["train"],
			support_batch_size=support_batch_size,
			query_input_sequence=seqs["val"],
			query_lengths=lengths["val"],
			query_batch_size=query_batch_size,
			epochs=end_epoch,
			init_epoch=init_epoch
		)
		model_file = "epoch-{}.maml".format(end_epoch)
		model_path = mconf.model_save_dir_prefix + model_file
		net.save_model(model_path)
		mconf.last_maml_ckpt = model_file
		print("evaluation\n-------")
		for t in range(mconf.num_tasks):
			print("inferring task {} ...".format(t+1))
			for s in [0, 1]:
				inferred_seqs = net.infer(
					seqs["val"][t][s], lengths["val"][t][s],
					src=s, tgt=1-s
				)
				sents = vocab.decode_sents(inferred_seqs)
				with open(mconf.output_dir_prefix + "epoch-{}_t{}_{}-{}.maml".format(end_epoch, t+1, s, 1-s), 'w', encoding="utf-8") as f:
					for sent in sents:
						f.write(sent + '\n')
				print("\t{}: ".format(s), inferred_seqs.shape)


def _fine_tune(net, mconf, seqs, lengths, vocab, total_epochs=6, epochs_per_val=2, batch_size=64, task_id=1):

	print("transfer learning ...")
	turns = total_epochs // epochs_per_val
	if total_epochs % epochs_per_val:
		turns += 1

	for turn in range(turns):
		init_epoch = turn * epochs_per_val
		end_epoch = min(total_epochs, (turn + 1) * epochs_per_val)
		net.fine_tune(
			input_sequence_all=seqs["train"], 
			lengths_all=lengths["train"],
			batch_size=batch_size,
			epochs=end_epoch,
			init_epoch=init_epoch
		)
		model_file = "epoch-{}.t{}".format(end_epoch, task_id)
		model_path = mconf.model_save_dir_prefix + model_file
		net.save_model(model_path)
		mconf.last_tsf_ckpts["t{}".format(task_id)] = model_file
		print("evaluation\n-------")
		print("inferring ...")
		for s in [0, 1]:
			inferred_seqs = net.infer(
				seqs["val"][s], lengths["val"][s],
				src=s, tgt=1-s
			)
			sents = vocab.decode_sents(inferred_seqs)
			with open(mconf.output_dir_prefix + "epoch-{}_t{}_{}-{}.transfer".format(end_epoch, task_id, s, 1-s), 'w', encoding="utf-8") as f:
				for sent in sents:
					f.write(sent + '\n')
			print("\t{}: ".format(s), inferred_seqs.shape)


def run_maml(mconf, device, load_data=False, load_model=False, maml_epochs=10, transfer_epochs=6, epochs_per_val=2, infer_task='', maml_batch_size=8, sub_batch_size=32, train_batch_size=64, from_pretrain=False):

	if maml_epochs > 0 or transfer_epochs > 0:
		print("loading data ...")
		vocab, seqs, lengths = utils.data_loader.load_data(mconf=mconf, load_data=load_data, save=(not load_data))
		utils.data_loader.print_data_info(vocab, seqs, lengths, mconf.num_tasks)
	else:
		print("inference mode")
		if from_pretrain:
			with open(mconf.processed_data_save_dir_prefix + "pretrain/vocab".format(mconf.num_tasks), "rb") as f:
				vocab = pickle.load(f)
		else:
			with open(mconf.processed_data_save_dir_prefix + "{}t/vocab".format(mconf.num_tasks), "rb") as f:
				vocab = pickle.load(f)

	printer = pprint.PrettyPrinter(indent=4)
	print(">>>>>>> Model Config <<<<<<<")
	printer.pprint(vars(mconf))

	if mconf.wordvec_path is None:
		init_embedding = None
	else:
		print("loading initial embedding from {} ...".format(mconf.wordvec_path))
		init_embedding = utils.data_loader.load_embedding_from_wdv(vocab, mconf.wordvec_path)
		mconf.embedding_size = init_embedding.shape[1]

	net = models.maml_cross_align.MAMLCrossAlign(
		device=device, num_tasks=mconf.num_tasks, 
		init_embedding=init_embedding, mconf=mconf
	)
	if load_model:
		net.load_model(mconf.model_save_dir_prefix + mconf.last_ckpt)

	# meta training
	if maml_epochs > 0:
		# the all tasks are used for maml training, and the specified tasks are required for transfer learning
		maml_seqs = {
			"train": seqs["train"],
			"val": seqs["val"]
		}
		maml_lengths = {
			"train": lengths["train"],
			"val": lengths["val"]
		}
		_train_maml(
			net, mconf, maml_seqs, maml_lengths, vocab=vocab,
			total_epochs=maml_epochs, epochs_per_val=epochs_per_val,
			support_batch_size=sub_batch_size, query_batch_size=maml_batch_size
		)

		model_file = "epoch-{}.maml".format(maml_epochs)
		model_path = mconf.model_save_dir_prefix + model_file
		net.save_model(model_path)
		mconf.last_ckpt = model_file
		mconf.last_maml_ckpt = model_file
	if transfer_epochs > 0:
		for t in mconf.tsf_tasks:
			# reset to the maml ckpt
			net.load_model(mconf.model_save_dir_prefix + mconf.last_maml_ckpt)
			transfer_seqs = {
				"train": seqs["train"][t-1],
				"val": seqs["val"][t-1]
			}
			transfer_lengths = {
				"train": lengths["train"][t-1],
				"val": lengths["val"][t-1]
			}
			_fine_tune(
				net, mconf, transfer_seqs, transfer_lengths, vocab=vocab,
				total_epochs=transfer_epochs, epochs_per_val=epochs_per_val,
				batch_size=train_batch_size, task_id=t
			)
			model_file = "epoch-{}.t{}".format(transfer_epochs, t)
			model_path = mconf.model_save_dir_prefix + model_file
			net.save_model(model_path)
			mconf.last_ckpt = model_file
			mconf.last_tsf_ckpts["t{}".format(t)] = model_file
	
	'''
	if maml_epochs > 0 or transfer_epochs > 0:
		model_file = dt.datetime.now().strftime("%Y%m%d%H%M")
		model_path = mconf.model_save_dir_prefix + model_file
		net.save_model(model_path)
		mconf.last_ckpt = model_file
	'''

	if infer_task != '':
		infer_task = int(infer_task)
		net.load_model(mconf.model_save_dir_prefix + mconf.last_tsf_ckpts["t{}".format(infer_task)])
		s0, s1, l0, l1 = utils.data_processor.load_task_data(
			infer_task, mconf.data_dir_prefix, vocab,
			label="infer", mconf=mconf
		)
		infer_seqs = [s0, s1]
		infer_lengths = [l0, l1]
		for s in [0, 1]:
			inferred_seqs = net.infer(
				infer_seqs[s], infer_lengths[s],
				src=s, tgt=1-s
			)
			sents = vocab.decode_sents(inferred_seqs)
			with open(mconf.output_dir_prefix + "infer_t{}_{}-{}".format(infer_task, s, 1-s), 'w', encoding="utf-8") as f:
				for sent in sents:
					f.write(sent + '\n')


	return net


def run_online_inference(mconf, ckpt, device):

	net = models.maml_cross_align.MAMLCrossAlign(
		device=device, num_tasks=mconf.num_tasks, 
		mconf=mconf
	)
	model_path = mconf.model_save_dir_prefix + ckpt
	net.load_model(model_path)

	with open(mconf.processed_data_save_dir_prefix + "{}t/vocab".format(mconf.num_tasks), "rb") as f:
		vocab = pickle.load(f)

	while True:
		sys.stdout.write("> ")
		sys.stdout.flush()

		cmd = sys.stdin.readline().rstrip()
		if cmd in ["quit", "exit"]:
			print("exiting ...")
			break
		seq = np.array(vocab.encode_sents([cmd], length=mconf.max_seq_length, pad_token=False), dtype="int32")
		length = utils.data_processor.get_sequence_lengths(seq, mconf.min_seq_length, mconf.max_seq_length)
		for s in [0, 1]:
			tsf = net.infer(
				seq, length,
				src=s, tgt=1-s
			)
			tsf = vocab.decode_sents(tsf)
			print("[{}->{}]: {}".format(s, 1-s, tsf[0]))



def extract_embeddings(mconf, ckpt, task_id, device, pretrain=False, sample_size=1000):

	net = models.maml_cross_align.MAMLCrossAlign(
		device=device, num_tasks=mconf.num_tasks,
		mconf=mconf
	)
	model_path = mconf.model_save_dir_prefix + ckpt
	net.load_model(model_path)

	if pretrain:
		dir_prefix = mconf.processed_data_save_dir_prefix + "pretrain/"
	else:
		dir_prefix = mconf.processed_data_save_dir_prefix + "{}t/".format(mconf.num_tasks)

	with open(dir_prefix + "vocab", "rb") as f:
		vocab = pickle.load(f)

	with open(dir_prefix + "t{}.val".format(task_id), "rb") as f:
		data = pickle.load(f)
		s0, s1 = data["s0"], data["s1"]
		l0, l1 = data["l0"], data["l1"]

		inds0 = np.random.choice(list(range(l0.shape[0])), sample_size)
		inds1 = np.random.choice(list(range(l1.shape[0])), sample_size)

		style_embeddings, content_embeddings = net.get_batch_embeddings(
			np.concatenate((s0, s1), axis=0),
			np.concatenate((l0, l1), axis=0)
		)
		style_embeddings = style_embeddings.cpu().detach().numpy()
		content_embeddings = content_embeddings.cpu().detach().numpy()

		with open(mconf.emb_save_dir_prefix + "t{}/extract.emb".format(task_id), "wb") as f:
			pickle.dump({
				"style": [style_embeddings[:sample_size, :], style_embeddings[sample_size:, :]],
				"content": [content_embeddings[:sample_size, :], content_embeddings[sample_size:, :]]
			}, f)
			print("dumped embeddings to {}t{}/extract.emb".format(mconf.emb_save_dir_prefix, task_id))
