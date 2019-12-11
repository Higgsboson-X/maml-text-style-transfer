import pprint

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
# ----------------

import config.model_config

def train_wordvec(mconf, args):

	if args.train_text_path == '':
		if args.task_id == 0:
			# all files
			train_text_path = mconf.data_dir_prefix + "text.pretrain"
		else:
			train_text_path = mconf.data_dir_prefix + "t{}.all".format(args.task_id)
	else:
		train_text_path = args.train_text_path

	print("loading input file and training model using {} ...".format(train_text_path))
	wdv = Word2Vec(LineSentence(train_text_path), min_count=1, size=mconf.embedding_size)
	print("trained wdv model: {}".format(wdv))
	wdv.wv.save_word2vec_format(mconf.model_save_dir_prefix + "wordvec.{}".format(args.task_id), binary=False)
	print("saved wdv to {}".format(mconf.model_save_dir_prefix + "wordvec.{}".format(args.task_id)))

	mconf.wordvec_path = mconf.model_save_dir_prefix + "wordvec.{}".format(args.task_id)

