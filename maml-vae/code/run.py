import sys
import json
import pprint
import pickle
import torch
import numpy as np
import datetime as dt

# ----------------
import config.model_config
import models.vae
import models.maml_vae
import utils.data_loader
import utils.data_processor

def _train_maml(net, mconf, seqs, lengths, labels, bows, vocab, total_epochs=10, epochs_per_val=2, support_batch_size=32, query_batch_size=8):

	print("maml learning ...")
	turns = total_epochs // epochs_per_val
	if total_epochs % epochs_per_val:
		turns += 1

	for turn in range(turns):
		init_epoch = turn * epochs_per_val
		end_epoch = min(total_epochs, (turn + 1) * epochs_per_val)
		net.train_maml(
			support_input_sequences=seqs["train"], 
			support_lengths=lengths["train"], 
			support_labels=labels["train"], 
			support_bow_representations=bows["train"], 
			support_batch_size=support_batch_size, 
			query_input_sequences=seqs["val"], 
			query_lengths=lengths["val"], 
			query_labels=labels["val"], 
			query_bow_representations=bows["val"], 
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
			style_embeddings = net.get_batch_style_embeddings(
				input_sequences=np.concatenate(seqs["val"][t], axis=0),
				lengths=np.concatenate(lengths["val"][t], axis=0)
			)
			style_conditioning_embeddings = [
				torch.mean(style_embeddings[:lengths["val"][t][0].shape[0]], axis=0),
				torch.mean(style_embeddings[lengths["val"][t][0].shape[0]:], axis=0)
			]
			for s in [0, 1]:
				inferred_seqs, style_preds = net.infer(
					seqs["val"][t][s], lengths["val"][t][s],
					style_conditioning_embedding=style_conditioning_embeddings[1-s].cpu().detach().numpy()
				)
				sents = vocab.decode_sents(inferred_seqs)
				with open(mconf.output_dir_prefix + "epoch-{}_t{}_{}-{}.maml".format(end_epoch, t+1, s, 1-s), 'w', encoding="utf-8") as f:
					for sent, pred in zip(sents, style_preds):
						f.write(sent + '\t' + str(pred.item()) + '\n')
				print("\t{}: ".format(s), inferred_seqs.shape)


def _fine_tune(net, mconf, seqs, lengths, labels, bows, vocab, total_epochs=6, epochs_per_val=2, batch_size=64, task_id=1):

	print("transfer learning ...")
	turns = total_epochs // epochs_per_val
	if total_epochs % epochs_per_val:
		turns += 1

	for turn in range(turns):
		init_epoch = turn * epochs_per_val
		end_epoch = min(total_epochs, (turn + 1) * epochs_per_val)
		net.fine_tune(
			input_sequences_all=seqs["train"], 
			lengths_all=lengths["train"], 
			labels_all=labels["train"], 
			bow_representations_all=bows["train"], 
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
		style_embeddings = net.get_batch_style_embeddings(
			input_sequences=np.concatenate(seqs["val"], axis=0), 
			lengths=np.concatenate(lengths["val"], axis=0)
		)
		style_conditioning_embeddings = [
			torch.mean(style_embeddings[:lengths["val"][0].shape[0]], axis=0),
			torch.mean(style_embeddings[lengths["val"][0].shape[0]:], axis=0)
		]
		for s in [0, 1]:
			inferred_seqs, style_preds = net.infer(
				seqs["val"][s], lengths["val"][s],
				style_conditioning_embedding=style_conditioning_embeddings[1-s].cpu().detach().numpy()
			)
			sents = vocab.decode_sents(inferred_seqs)
			with open(mconf.output_dir_prefix + "epoch-{}_t{}_{}-{}.transfer".format(end_epoch, task_id, s, 1-s), 'w', encoding="utf-8") as f:
				for sent, pred in zip(sents, style_preds):
					f.write(sent + '\t' + str(pred.item()) + '\n')
			print("\t{}: ".format(s), inferred_seqs.shape)


def run_maml(mconf, device, load_data=False, load_model=False, maml_epochs=10, transfer_epochs=6, epochs_per_val=2, infer_task='', maml_batch_size=8, sub_batch_size=32, train_batch_size=64):

	if maml_epochs > 0 or transfer_epochs > 0:
		print("loading data ...")
		vocab, seqs, lengths, labels, bows = utils.data_loader.load_data(mconf=mconf, load_data=load_data, save=(not load_data))
		utils.data_loader.print_data_info(vocab, seqs, lengths, labels, bows, mconf.num_tasks)
	else:
		print("inference mode")
		with open(mconf.processed_data_save_dir_prefix + "{}t/vocab".format(mconf.num_tasks), "rb") as f:
			vocab = pickle.load(f)
		
		mconf.vocab_size = vocab._size
		mconf.bow_size = vocab._bows

	printer = pprint.PrettyPrinter(indent=4)
	print(">>>>>>> Model Config <<<<<<<")
	printer.pprint(vars(mconf))

	if mconf.wordvec_path is None:
		init_embedding = None
	else:
		print("loading initial embedding from {} ...".format(mconf.wordvec_path))
		init_embedding = utils.data_loader.load_embedding_from_wdv(vocab, mconf.wordvec_path)
		mconf.embedding_size = init_embedding.shape[1]

	net = models.maml_vae.MAMLAdvAutoencoder(
		device=device, num_tasks=mconf.num_tasks, 
		init_embedding=init_embedding, mconf=mconf
	)
	if load_model:
		net.load_model(mconf.model_save_dir_prefix + mconf.last_ckpt)

	# meta training
	if maml_epochs > 0:
		# use all tasks for maml learning, and specified tasks for fine-tuning
		maml_seqs = {
			"train": seqs["train"],
			"val": seqs["val"]
		}
		maml_lengths = {
			"train": lengths["train"],
			"val": lengths["val"]
		}
		maml_labels = {
			"train": labels["train"],
			"val": labels["val"]
		}
		maml_bows = {
			"train": bows["train"],
			"val": bows["val"]
		}
		_train_maml(
			net, mconf, maml_seqs, maml_lengths, maml_labels, maml_bows, vocab=vocab,
			total_epochs=maml_epochs, epochs_per_val=epochs_per_val,
			support_batch_size=sub_batch_size, query_batch_size=maml_batch_size
		)
	if transfer_epochs > 0:
		for t in mconf.tsf_tasks:
			transfer_seqs = {
			"train": seqs["train"][t-1],
			"val": seqs["val"][t-1]
			}
			transfer_lengths = {
				"train": lengths["train"][t-1],
				"val": lengths["val"][t-1]
			}
			transfer_labels = {
				"train": labels["train"][t-1],
				"val": labels["val"][t-1]
			}
			transfer_bows = {
				"train": bows["train"][t-1],
				"val": bows["val"][t-1]
			}
			_fine_tune(
				net, mconf, transfer_seqs, transfer_lengths, transfer_labels, transfer_bows, vocab=vocab,
				total_epochs=transfer_epochs, epochs_per_val=epochs_per_val,
				batch_size=train_batch_size, task_id=t
			)
	
	if maml_epochs > 0 or transfer_epochs > 0:
		model_file = dt.datetime.now().strftime("%Y%m%d%H%M")
		model_path = mconf.model_save_dir_prefix + model_file
		net.save_model(model_path)
		mconf.last_ckpt = model_file

	if infer_task != '':
		infer_task = int(infer_task)
		net.load_model(mconf.model_save_dir_prefix + mconf.last_tsf_ckpts["t{}".format(infer_task)])
		s0, s1, l0, l1, lb0, lb1, bow0, bow1 = utils.data_processor.load_task_data(
			infer_task, mconf.data_dir_prefix, vocab,
			label="infer", mconf=mconf
		)
		infer_seqs = [s0, s1]
		infer_lengths = [l0, l1]
		infer_labels = [lb0, lb1]
		infer_bows = [bow0, bow1]
		style_embeddings = net.get_batch_style_embeddings(
			input_sequences=np.concatenate(infer_seqs, axis=0), 
			lengths=np.concatenate(infer_lengths, axis=0)
		)
		style_conditioning_embeddings = [
			torch.mean(style_embeddings[:infer_lengths[0].shape[0]], axis=0),
			torch.mean(style_embeddings[infer_lengths[0].shape[0]:], axis=0)
		]
		for s in [0, 1]:
			inferred_seqs, style_preds = net.infer(
				infer_seqs[s], infer_lengths[s],
				style_conditioning_embedding=style_conditioning_embeddings[1-s].cpu().detach().numpy()
			)
			sents = vocab.decode_sents(inferred_seqs)
			with open(mconf.output_dir_prefix + "infer_t{}_{}-{}".format(infer_task, s, 1-s), 'w', encoding="utf-8") as f:
				for sent, pred in zip(sents, style_preds):
					f.write(sent + '\t' + str(pred.item()) + '\n')

	return net


def run_online_inference(mconf, ckpt, tgt_file, device):

	with open(mconf.processed_data_save_dir_prefix + "{}t/vocab".format(mconf.num_tasks), "rb") as f:
		vocab = pickle.load(f)

	seqs, lengths, _, _ = utils.data_processor.get_seq_data_from_file(
		filename=tgt_file, vocab=vocab, mconf=mconf
	)

	net = models.maml_vae.MAMLAdvAutoencoder(
		device=device, num_tasks=mconf.num_tasks, 
		mconf=mconf
	)
	model_path = mconf.model_save_dir_prefix + ckpt
	net.load_model(model_path)

	style_embeddings = net.get_batch_style_embeddings(
		input_sequences=seqs, 
		lengths=lengths
	)
	style_conditioning_embedding = torch.mean(style_embeddings, axis=0)

	while True:
		sys.stdout.write("> ")
		sys.stdout.flush()

		cmd = sys.stdin.readline().rstrip()
		if cmd in ["quit", "exit"]:
			print("exiting ...")
			break
		seq = np.array(vocab.encode_sents([cmd], length=mconf.max_seq_length, pad_token=False), dtype="int32")
		length = utils.data_processor.get_sequence_lengths(seq, mconf.min_seq_length, mconf.max_seq_length)
		tsf, pred = net.infer(
			seq, length,
			style_conditioning_embedding=style_conditioning_embedding.cpu().detach().numpy()
		)
		tsf = vocab.decode_sents(tsf)
		print("[tsf]: {} (pred = {})".format(tsf[0], pred[0].item()))

