import sys
import os
import json
import pprint
import pickle
import torch
import argparse
import numpy as np
import datetime as dt

# ----------------
import config.model_config
import models.cross_align
import utils.data_loader
import utils.data_processor
import utils.vocab


def load_args():

	parser = argparse.ArgumentParser(
		prog="CROSS_ALIGN", 
		description="CrossAlign Text Style Transfer Model"
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
		"--task-id", type=str, default='',
		help="task id for training"
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
		"--batch-size", type=int, default=64,
		help="batch size for fine-tuning"
	)

	parser.add_argument(
		"--epochs", type=int, default=10,
		help="training epochs"
	)
	parser.add_argument(
		"--pretrain-epochs", type=int, default=10,
		help="seq2seq pretraining epochs"
	)
	
	parser.add_argument(
		"--epochs-per-val", type=int, default=2,
		help="epochs per validation and checkpointing"
	)
	
	parser.add_argument(
		"--online-inference", action="store_true",
		help="whether to do online inference, suppressing other arguments"
	)
	parser.add_argument(
		"--ckpt", type=str, default="final",
		help="checkpoint to recover model, should be provided in online inference mode"
	)

	# device
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
		mconf = config.model_config.default_maml_mconf

	return mconf


def build_mconf_from_args(args):

	mconf = config.model_config.ModelConfig()

	for attr in vars(mconf).keys():
		if hasattr(args, attr):
			setattr(mconf, attr, getattr(args, attr))

	mconf.update_corpus()

	return mconf


# ======================================================================================
def run_online_inference(mconf, ckpt, device, task_id):

	net = models.cross_align.CrossAlign(device=device, mconf=mconf)
	model_path = mconf.model_save_dir_prefix + ckpt
	net.load_model(model_path)

	with open(mconf.processed_data_save_dir_prefix + "1t/t{}.vocab".format(task_id), "rb") as f:
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


def run_cross_align(mconf, device, load_data=False, load_model=False, epochs=20, pretrain_epochs=10, epochs_per_val=5, batch_size=64, task_id=1):

	# --------------------------------------------------------------
	if pretrain_epochs > 0:
		# pretrain file should follow the required format
		if os.path.exists(mconf.processed_data_save_dir_prefix + "pretrain"):
			print("loading processed pretrain data from {} ...".format(mconf.processed_data_save_dir_prefix + "pretrain"))
			with open(mconf.processed_data_save_dir_prefix + "pretrain/vocab", "rb") as f:
				vocab = pickle.load(f)
			with open(mconf.processed_data_save_dir_prefix + "pretrain/data", "rb") as f:
				data = pickle.load(f)
				pretrain_seqs, pretrain_lengths = data["seqs"], data["lengths"]
		else:
			pretrain_file_path = mconf.data_dir_prefix + "text.pretrain"
			os.makedirs(mconf.processed_data_save_dir_prefix + "pretrain")

			vocab = utils.vocab.Vocabulary(mconf=mconf)
			print("updating vocab from {} ...".format(pretrain_file_path))
			vocab.update_vocab(pretrain_file_path)

			with open(mconf.processed_data_save_dir_prefix + "pretrain/vocab", "wb") as f:
				pickle.dump(vocab, f)

			print("loading pretrain data from {} ...".format(pretrain_file_path))
			
			pretrain_seqs, pretrain_lengths = utils.data_processor.get_seq_data_from_file(pretrain_file_path, vocab, mconf)
			data = {"seqs": pretrain_seqs, "lengths": pretrain_lengths}
			with open(mconf.processed_data_save_dir_prefix + "pretrain/data", "wb") as f:
				pickle.dump(data, f)

			print("saved processed pretrain data to {}".format(mconf.processed_data_save_dir_prefix + "pretrain"))

		mconf.vocab_size = vocab._size

	if epochs > 0 or pretrain_epochs > 0:
		# pretrain mode needs the val data
		if pretrain_epochs > 0:
			# pretrain preprocessing should have been done above
			dir_prefix = mconf.processed_data_save_dir_prefix + "pretrain/"
		else:
			dir_prefix = mconf.processed_data_save_dir_prefix + "1t/"

		print("loading data ...")
		if load_data:
			if pretrain_epochs <= 0:
				# should never go there in pretrain mode
				with open(dir_prefix + "t{}.vocab".format(task_id), "rb") as f:
					vocab = pickle.load(f)
			# either in pretrain dir or 1t dir
			seqs, lengths = {}, {}
			for label in ["train", "val"]:
				with open(dir_prefix + "t{}.{}".format(task_id, label), "rb") as f:
					data = pickle.load(f)
					s0, s1 = data["s0"], data["s1"]
					l0, l1 = data["l0"], data["l1"]
					seqs[label] = [s0, s1]
					lengths[label] = [l0, l1]
		else:
			if pretrain_epochs <= 0:
				# non-pretrain mode
				vocab = utils.vocab.Vocabulary(mconf=mconf)
				print("updating vocab from task {} ...".format(task_id))
				for s in [0, 1]:
					vocab.update_vocab(mconf.data_dir_prefix + "train/t{}.{}".format(task_id, s))
					vocab.update_vocab(mconf.data_dir_prefix + "val/t{}.{}".format(task_id, s))
				with open(dir_prefix + "t{}.vocab".format(task_id), "wb") as f:
					pickle.dump(vocab, f)
			
			seqs, lengths = {}, {}
			for label in ["train", "val"]:
				s0, s1, l0, l1 = utils.data_processor.load_task_data(
					task_id, mconf.data_dir_prefix, vocab, 
					label=label, mconf=mconf
				)
				seqs[label] = [s0, s1]
				lengths[label] = [l0, l1]

			data_train, data_val = dict(), dict()
			for s in [0, 1]:
				data_train["s{}".format(s)] = seqs["train"][s]
				data_train["l{}".format(s)] = lengths["train"][s]
				data_val["s{}".format(s)] = seqs["val"][s]
				data_val["l{}".format(s)] = lengths["val"][s]

			with open(dir_prefix + "t{}.train".format(task_id), "wb") as f:
				pickle.dump(data_train, f)
			with open(dir_prefix + "t{}.val".format(task_id), "wb") as f:
				pickle.dump(data_val, f)

			print("saved data for task {}".format(task_id))

		mconf.vocab_size = vocab._size

		print("vocab_size = {}".format(vocab._size))
		for key in ["train", "val"]:
			print("{}\n-------".format(key))
			print("task {} data:".format(task_id))
			for s in [0, 1]:
				print("\t {}:".format(s), seqs[key][s].shape, lengths[key][s].shape)

	if pretrain_epochs > 0:
		# make pretrain_seqs the same format
		midpoint = int(pretrain_seqs.shape[0]/2)
		pretrain_seqs = {"train": [pretrain_seqs[:midpoint], pretrain_seqs[midpoint:]], "val": seqs["val"]}
		pretrain_lengths = {"train": [pretrain_lengths[:midpoint], pretrain_lengths[midpoint:]], "val": lengths["val"]}

	if pretrain_epochs <= 0 and epochs <= 0:
		print("no operations, exiting ...")
		return

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

	net = models.cross_align.CrossAlign(device=device, init_embedding=init_embedding, mconf=mconf)
	if load_model:
		net.load_model(mconf.model_save_dir_prefix + mconf.last_ckpt)

	if pretrain_epochs > 0:
		print("pretraining seq2seq ...")
		_train_cross_align(
			net, mconf, pretrain_seqs, pretrain_lengths, vocab=vocab,
			total_epochs=pretrain_epochs, epochs_per_val=epochs_per_val,
			batch_size=batch_size, task_id=task_id, pretrain=True
		)

		model_file = "pretrain-{}.t{}".format(pretrain_epochs, task_id)
		model_path = mconf.model_save_dir_prefix + model_file
		net.save_model(model_path)
		mconf.last_ckpt = model_file

	if epochs > 0:
		print("training cross_align ...")
		_train_cross_align(
			net, mconf, seqs, lengths, vocab=vocab,
			total_epochs=epochs, epochs_per_val=epochs_per_val,
			batch_size=batch_size, task_id=task_id, pretrain=False
		)

		model_file = "train-{}.t{}".format(epochs, task_id)
		model_path = mconf.model_save_dir_prefix + model_file
		net.save_model(model_path)
		mconf.last_ckpt = model_file

	# model_file = dt.datetime.now().strftime("%Y%m%d%H%M")
	# model_path = mconf.model_save_dir_prefix + model_file
	# net.save_model(model_path)
	# mconf.last_ckpt = model_file

	return net


def _train_cross_align(net, mconf, seqs, lengths, vocab, total_epochs, epochs_per_val, batch_size, task_id, pretrain=False):

	print("training cross_align ...")
	print("pretrain mode: ", pretrain)
	turns = total_epochs // epochs_per_val
	if total_epochs % epochs_per_val:
		turns += 1

	for turn in range(turns):
		init_epoch = turn * epochs_per_val
		end_epoch = min(total_epochs, (turn + 1) * epochs_per_val)
		net.train(
			input_sequence_all=seqs["train"], 
			lengths_all=lengths["train"],
			batch_size=batch_size,
			epochs=end_epoch,
			init_epoch=init_epoch, pretrain=pretrain
		)
		if pretrain:
			model_file = "pretrain-{}.t{}".format(end_epoch, task_id)
		else:
			model_file = "train-{}.t{}".format(end_epoch, task_id)
		model_path = mconf.model_save_dir_prefix + model_file
		net.save_model(model_path)
		mconf.last_ckpt = model_file
		print("evaluation\n-------")
		print("inferring ...")
		for s in [0, 1]:
			inferred_seqs = net.infer(
				seqs["val"][s], lengths["val"][s],
				src=s, tgt=1-s
			)
			sents = vocab.decode_sents(inferred_seqs)
			if pretrain:
				output_file = "pretrain-{}_t{}_{}-{}".format(end_epoch, task_id, s, 1-s)
			else:
				output_file = "train-{}_t{}_{}-{}".format(end_epoch, task_id, s, 1-s)
			with open(mconf.output_dir_prefix + output_file, 'w', encoding="utf-8") as f:
				for sent in sents:
					f.write(sent + '\n')
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
			device=torch.device("cpu"), task_id=args.task_id
		)
	else:
		if not args.disable_gpu and torch.cuda.is_available():
			device = torch.device("cuda:{}".format(args.device_index))
		else:
			device = torch.device("cpu")
		print("[DEVICE INFO] using {}".format(device))

		run_cross_align(
			mconf=mconf, device=device, load_data=args.load_data, load_model=args.load_model,
			epochs=args.epochs, pretrain_epochs=args.pretrain_epochs, epochs_per_val=args.epochs_per_val, 
			batch_size=args.batch_size, task_id=args.task_id
		)
		with open(mconf.output_dir_prefix + "t{}.json".format(args.task_id), 'w') as f:
			json.dump(vars(mconf), f, indent=4)
			print("saved model config to {}".format(mconf.output_dir_prefix + "{}.json".format(args.task_id)))

	print(">>>>>>> Completed <<<<<<<")


if __name__ == "__main__":

	main()