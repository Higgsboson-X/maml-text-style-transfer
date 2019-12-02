class ModelConfig(object):

	def __init__(self):

		self.dim_y = 32
		self.dim_z = 128
		self.dim_h = self.dim_y + self.dim_z

		self.num_labels = 2

		self.betas = [0.5, 0.999]
		self.grad_clip = 30.

		self.eta = 10.
		self.d_loss_tolerance = 1.2

		self.gamma_init = 0.001
		self.gamma_min = 0.0001
		self.gamma_decay = 0.9

		# not including pad
		self.min_seq_length = 5
		self.max_seq_length = 15

		self.vocab_size = 5000
		self.vocab_cutoff = 7

		self.pad_id = 0
		self.eos_id = 1
		self.sos_id = 2
		self.unk_id = 3

		self.filter_stopwords = False
		self.embedding_size = 100

		self.n_filters = 32
		self.filter_sizes = [2, 3, 4, 5]

		self.dropout = 0.1

		self.train_lr = 0.01 # fine-tuning

		self.corpus = "translations"

		self.adv_loss_weight = 0.7

		self.last_ckpt = "final"

		self.model_save_dir_prefix = "../ckpt/{}/".format(self.corpus)
		self.processed_data_save_dir_prefix = "../data/{}/processed/".format(self.corpus)
		self.data_dir_prefix = "../data/{}/".format(self.corpus)

		self.output_dir_prefix = "../output/{}/".format(self.corpus)

		self.wordvec_path = None

	def init_from_dict(self, config):

		for key in config:
			setattr(self, key, config[key])

	def update_corpus(self):

		self.model_save_dir_prefix = "../ckpt/{}/".format(self.corpus)
		self.processed_data_save_dir_prefix = "../data/{}/processed/".format(self.corpus)
		self.data_dir_prefix = "../data/{}/".format(self.corpus)

		self.output_dir_prefix = "../output/{}/".format(self.corpus)


class MAMLModelConfig(ModelConfig):

	def __init__(self):

		super(MAMLModelConfig, self).__init__()

		self.meta_lr = 0.001 # meta-learner
		self.sub_lr = 0.01 # subtask

		self.num_updates = 2

		self.num_tasks = 7

		# self.processed_data_save_dir_prefix = "../data/{}/processed/{}t/".format(self.corpus, self.num_tasks)

	'''
	def update_corpus(self):

		super(MAMLModelConfig, self).update_corpus()
		self.processed_data_save_dir_prefix = "../data/{}/processed/{}t/".format(self.corpus, self.num_tasks)
	'''

default_mconf = ModelConfig()
default_maml_mconf = MAMLModelConfig()
