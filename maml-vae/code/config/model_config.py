class ModelConfig(object):

	def __init__(self):

		self.num_labels = 2

		self.betas = [0.5, 0.999]
		self.grad_clip = 30.

		self.adv_loss_tolerance = 7.

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

		self.embedding_size = 300

		self.encoder_rnn_size = 128
		self.decoder_rnn_size = 128

		self.sentence_embedding_size = 2 * self.encoder_rnn_size

		self.style_embedding_size = 8
		self.content_embedding_size = self.encoder_rnn_size - self.style_embedding_size

		# self.bow_size = 5000 # filtered stopwords and punctuations

		self.sequence_word_dropout = 0.1
		self.rnn_dropout = 0.1
		self.fc_dropout = 0.1

		self.kl_anneal_iterations = 2000
		self.eps = 1e-8

		# fine-tuning lr
		self.autoencoder_train_lr = 0.001
		self.adversarial_train_lr = 0.001
		self.style_overall_train_lr = 0.001

		self.style_kl_weight = 0.03
		self.content_kl_weight = 0.03

		self.style_adv_loss_weight = 1
		self.content_adv_loss_weight = 0.03
		self.style_multitask_loss_weight = 10
		self.content_multitask_loss_weight = 3

		self.corpus = "yelp"

		self.last_ckpt = "final"

		self.wordvec_path = None

		# --------------------------------------------------------------
		# requires corpus update

		self.model_save_dir_prefix = "../ckpt/{}/".format(self.corpus)
		self.processed_data_save_dir_prefix = "../data/{}/processed/".format(self.corpus)
		self.data_dir_prefix = "../data/{}/".format(self.corpus)

		self.output_dir_prefix = "../output/{}/".format(self.corpus)

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

		self.meta_autoencoder_lr = 0.001
		self.meta_adversarial_lr = 0.001
		self.meta_style_overall_lr = 0.01

		self.sub_autoencoder_lr = 0.001
		self.sub_adversarial_lr = 0.001
		self.sub_style_overall_lr = 0.01

		self.num_updates = 2

		self.num_tasks = 3

		# self.processed_data_save_dir_prefix = "../data/{}/processed/{}t/".format(self.corpus, self.num_tasks)

	'''
	def update_corpus(self):

		super(MAMLModelConfig, self).update_corpus()
		self.processed_data_save_dir_prefix = "../data/{}/processed/{}t/".format(self.corpus, self.num_tasks)
	'''

default_mconf = ModelConfig()
default_maml_mconf = MAMLModelConfig()