import sys
import json
import pprint
import pickle
import torch
import argparse
import numpy as np
import datetime as dt

# ----------------
import config.model_config
import models.vae
import utils.data_loader
import utils.data_processor
import utils.vocab

def load_args():

	parser = argparse.ArgumentParser(
		prog="VAE",
		description="Adversarial-Autoencoder for Text Style Transfer"
	)

	parser.add_argument(
		"--config-path", type=str, default='',
		help="path for model configuration"
	)
	parser.add_argument(
		"--corpus", type=str, default="s1",
		help="training corpus name"
	)
	parser.add_argument(
		"--task-id", type=int, default=1,
		help="task id for training, (even if not in maml mode)"
	)
	parser.add_argument(
		"--load-data", action="store_true",
		help="whether to load processed data"
	)
	parser.add_argument(
		"--load-model", action="store_true", 
		help="whether to load model from last checkpoint"
	)

	parser.add_argument(
		"--batch-size", type=int, default=128,
		help="batch size for training"
	)
	parser.add_argument(
		"--epochs", type=int, default=20,
		help="training epochs"
	)
	parser.add_argument(
		"--epochs-per-val", type=int, default=5,
		help="epochs per validation and checkpointing"
	)
	parser.add_argument(
		"--inference", action="store_true",
		help="whether to perform inference"
	)

	parser.add_argument(
		"--online-inference", action="store_true",
		help="whether to do online inference, suppressing other arguments"
	)
	parser.add_argument(
		"--ckpt", type=str, default="final",
		help="checkpoint to recover model, should be provided in online inference mode"
	)
	parser.add_argument(
		"--tgt-file", type=str, default="./tgt",
		help="file with sentences in target style for online inference"
	)

	parser.add_argument(
		"--disable-gpu", action="store_true",
		help="whether to disable GPU usage"
	)
	parser.add_argument(
		"--device-index", type=int, default=0,
		help="GPU device index to use"
	)

	args = parser.parse_args()

	return args

def load_mconf(config_path=None):

	print("loading model config ...")
	if config_path is not None:
		with open(config_path, 'r') as f:
			config_json = json.load(f)
			mconf = config.model_config.ModelConfig()
			mconf.init_from_dict(config_json)
	else:
		# load default
		mconf = config.model_config.default_mconf

	return mconf


def build_mconf_from_args(args):

	mconf = config.model_config.ModelConfig()

	for attr in vars(mconf).keys():
		if hasattr(args, attr):
			setattr(mconf, attr, getattr(args, attr))

	mconf.update_corpus()

	return mconf


# ======================================================================================
def run_online_inference(mconf, ckpt, tgt_file, device, task_id):

	with open(mconf.processed_data_save_dir_prefix + "1t/t{}.vocab".format(task_id), "rb") as f:
		vocab = pickle.load()

	seqs, lengths, _, _ = utils.data_processor.get_seq_data_from_file(
		filename=tgt_file, vocab=vocab, mconf=mconf
	)

	net = models.vae.AdvAutoencoder(device=device, mconf=mconf)
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



def run_vae(mconf, device, load_data=False, load_model=False, epochs=20, epochs_per_val=5, inference=False, batch_size=64, task_id=1):

	# --------------------------------------------------------------
	if epochs > 0:
		
		print("loading data ...")
		if load_data:
			with open(mconf.processed_data_save_dir_prefix + "1t/t{}.vocab".format(task_id), "rb") as f:
				vocab = pickle.load(f)
			seqs, lengths, labels, bow_representations = {}, {}, {}, {}
			for label in ["train", "val"]:
				with open(mconf.processed_data_save_dir_prefix + "1t/t{}.{}".format(task_id, label), "rb") as f:
					data = pickle.load(f)
					seqs[label] = [data["s0"], data["s1"]]
					lengths[label] = [data["l0"], data["l1"]]
					labels[label] = [data["lb0"], data["lb1"]]
					bow_representations[label] = [data["bow0"], data["bow1"]]
		else:
			vocab = utils.vocab.Vocabulary(mconf=mconf)
			print("updating vocab from task {} ...".format(task_id))
			for s in [0, 1]:
				vocab.update_vocab(mconf.data_dir_prefix + "train/t{}.{}".format(task_id, s))
				vocab.update_vocab(mconf.data_dir_prefix + "val/t{}.{}".format(task_id, s))

			seqs, lengths, labels, bow_representations = {}, {}, {}, {}
			for label in ["train", "val"]:
				s0, s1, l0, l1, lb0, lb1, bow0, bow1 = utils.data_processor.load_task_data(
					task_id, mconf.data_dir_prefix, vocab,
					label=label, mconf=mconf
				)
				seqs[label] = [s0, s1]
				lengths[label] = [l0, l1]
				labels[label] = [lb0, lb1]
				bow_representations[label] = [bow0, bow1]

			with open(mconf.processed_data_save_dir_prefix + "1t/t{}.vocab".format(task_id), "wb") as f:
				pickle.dump(vocab, f)
			data_train, data_val = dict(), dict()

			for s in [0, 1]:
				data_train["s{}".format(s)] = seqs["train"][s]
				data_train["l{}".format(s)] = lengths["train"][s]
				data_train["lb{}".format(s)] = labels["train"][s]
				data_train["bow{}".format(s)] = bow_representations["train"][s]
				
				data_val["s{}".format(s)] = seqs["val"][s]
				data_val["l{}".format(s)] = lengths["val"][s]
				data_val["lb{}".format(s)] = labels["val"][s]
				data_val["bow{}".format(s)] = bow_representations["val"][s]

			with open(mconf.processed_data_save_dir_prefix + "1t/t{}.train".format(task_id), "wb") as f:
				pickle.dump(data_train, f)
			with open(mconf.processed_data_save_dir_prefix + "1t/t{}.val".format(task_id), "wb") as f:
				pickle.dump(data_val, f)

			print("saved data for task {}".format(task_id))

		mconf.vocab_size = vocab._size
		mconf.bow_size = vocab._bows

		for key in ["train", "val"]:
			print("{}\n-------".format(key))
			print("task {} data:".format(task_id))
			for s in [0, 1]:
				print(
					"\t {}:".format(s), seqs[key][s].shape, lengths[key][s].shape, 
					labels[key][s].shape, bow_representations[key][s].shape
				)

	else:
		print("inference mode")
		with open(mconf.processed_data_save_dir_prefix + "1t/t{}.vocab".format(task_id), "rb") as f:
			vocab = pickle.load(f)

		mconf.vocab_size = vocab._size
		mconf.bow_size = vocab._bows

	# --------------------------------------------------------------
	printer = pprint.PrettyPrinter(indent=4)
	print(">>>>>>> Model Config <<<<<<<")
	printer.pprint(vars(mconf))

	if mconf.wordvec_path is None:
		init_embedding = None
	else:
		print("loading initial embedding from {} ...".format(mconf.wordvec_path))
		init_embedding = utils.data_loader.load_embedding_from_wdv(vocab, mconf.wordvec_path)
		mconf.embedding_size = init_embedding.shape[1]

	net = models.vae.AdvAutoencoder(
		device=device, init_embedding=init_embedding,
		mconf=mconf
	)
	if load_model:
		net.load_model(mconf.model_save_dir_prefix + mconf.last_ckpt)

	if epochs > 0:
		_train_vae(
			net, mconf, seqs, lengths, labels, bow_representations, vocab=vocab,
			total_epochs=epochs, epochs_per_val=epochs_per_val,
			batch_size=batch_size, task_id=task_id
		)

		model_file = dt.datetime.now().strftime("%Y%m%d%H%M")
		model_path = mconf.model_save_dir_prefix + model_file
		net.save_model(model_path)
		mconf.last_ckpt = model_file

	if inference:
		if epochs <= 0:
			net.load_model(mconf.model_save_dir_prefix + mconf.last_ckpt)
		
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


def _train_vae(net, mconf, seqs, lengths, labels, bows, vocab, total_epochs=20, epochs_per_val=5, batch_size=64, task_id=1):

	print("training vae ...")
	turns = total_epochs // epochs_per_val
	if total_epochs % epochs_per_val:
		turns += 1

	for turn in range(turns):
		init_epoch = turn * epochs_per_val
		end_epoch = min(total_epochs, (turn + 1) * epochs_per_val)
		net.train(
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
		mconf.last_ckpt = model_file
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
			with open(mconf.output_dir_prefix + "epoch-{}_t{}_{}-{}".format(end_epoch, task_id, s, 1-s), 'w', encoding="utf-8") as f:
				for sent, pred in zip(sents, style_preds):
					f.write(sent + '\t' + str(pred.item()) + '\n')
			print("\t{}: ".format(s), inferred_seqs.shape)


# ======================================================================================

def main():

	args = load_args()

	printer = pprint.PrettyPrinter(indent=4)

	print(">>>>>>> Options <<<<<<<")
	printer.pprint(vars(args))

	if args.config_path != '':
		mconf = load_mconf(args.config_path)
	else:
		mconf = build_mconf_from_args(args)

	if args.online_inference:
		run_online_inference(
			mconf=mconf, ckpt=args.ckpt,
			tgt_file=args.tgt_file, device=torch.device("cpu"), task_id=args.task_id
		)
	else:
		if not args.disable_gpu and torch.cuda.is_available():
			device = torch.device("cuda:{}".format(args.device_index))
		else:
			device = torch.device("cpu")
		print("[DEVICE INFO] using {}".format(device))

		run_vae(
			mconf=mconf, device=device, load_data=args.load_data, load_model=args.load_model,
			epochs=args.epochs, epochs_per_val=args.epochs_per_val,
			batch_size=args.batch_size, task_id=args.task_id
		)
		with open(mconf.output_dir_prefix + "vae.json", 'w') as f:
			json.dump(vars(mconf), f, indent=4)
			print("saved model config to {}".format(mconf.output_dir_prefix + "vae.json"))

	print(">>>>>>> Completed <<<<<<<")


if __name__ == "__main__":

	main()



