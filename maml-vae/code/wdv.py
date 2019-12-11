import argparse
import pprint
import json

# ----------------

import config.model_config
import models.word2vec

def load_args():

	parser = argparse.ArgumentParser(
		prog="WORD2VEC",
		description="Gensim Word2Vec Model"
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
		"--train-text-path", type=str, default='',
		help="text file for training word vectors"
	)
	parser.add_argument(
		"--model", type=str, default="vae",
		help="model option: vae / maml"
	)
	parser.add_argument(
		"--task-id", type=int, default=0,
		help="task id for the corpus, 0 in maml mode"
	)
	# train using ../data/{corpus}/text.pretrain

	args = parser.parse_args()

	return args


def build_mconf_from_args(args):

	if args.model == "vae":
		mconf = config.model_config.ModelConfig()
	else:
		mconf = config.model_config.MAMLModelConfig()

	for attr in vars(mconf).keys():
		if hasattr(args, attr):
			setattr(mconf, attr, getattr(args, attr))

	mconf.update_corpus()

	return mconf



def load_mconf(args):

	print("loading model config ...")

	if args.model == "vae":
		mconf = config.model_config.ModelConfig()
	else:
		mconf = config.model_config.MAMLModelConfig()

	with open(args.config_path, 'r') as f:
		config_json = json.load(f)
		mconf.init_from_dict(config_json)

	return mconf


def main():

	args = load_args()

	printer = pprint.PrettyPrinter(indent=4)

	print(">>>>>>> Options <<<<<<<")
	printer.pprint(vars(args))

	if args.config_path != '':
		mconf = load_mconf(args)
	else:
		mconf = build_mconf_from_args(args)

	models.word2vec.train_wordvec(mconf, args)

	with open("../config/{}.json".format(args.corpus), 'w') as f:
		json.dump(vars(mconf), f, indent=4)
		print("saved model config to ../config/{}.json".format(args.corpus))

	print(">>>>>>> Completed <<<<<<<")


if __name__ == "__main__":

	main()