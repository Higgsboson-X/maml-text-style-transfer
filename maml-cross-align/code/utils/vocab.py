from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import pickle

from config.model_config import default_mconf

class Vocabulary:

	def __init__(self, mconf=default_mconf):

		self._word2id = {'<pad>': 0, '<eos>': 1, '<sos>': 2, '<unk>': 3}
		self._id2word = ['<pad>', '<eos>', '<sos>', '<unk>']
		self._tokenize = word_tokenize
		self._size = 4

		if mconf.filter_stopwords:
			self.filter_list = list(set(stopwords.words("english")))
		else:
			self.filter_list = []

		self.cutoff = mconf.vocab_cutoff

	def init_from_saved_vocab(self, path):

		f = open(path, 'rb')
		v = pickle.load(f)

		self._word2id = v._word2id
		self._id2word = v._id2word
		self._tokenize = v._tokenize
		self._size = v._size

		f.close()
		print("loaded vocabulary from {}".format(path))

	def update_vocab(self, path):

		f = open(path, 'r', encoding='utf-8')
		lines = f.readlines()
		print('lines: ' + str(len(lines)))

		added = 0
		sentences = 0
		words = {}

		for line in lines:
			sentences += 1
			line = line.lower()
			try:
				# in `label + \t + sent` format
				line = line.split('\t')[1]
			except:
				pass
			line = self._tokenize(line)
			if sentences % 5000 == 0:
				print("word: " + str(added) + ", lines: " + str(sentences))
			for word in line:
				if word in self.filter_list or word in self._word2id:
					continue
				if word not in words:
					words[word] = 1
					added += 1
				else:
					words[word] += 1
		added = 0
		for word in words:
			if words[word] >= self.cutoff and word not in self._word2id:
				self._word2id[word] = self._size + added
				self._id2word.append(word)
				added += 1

		print("updated " + str(added) + " words")

		self._size += added
		
		return self._size

	def word2id(self, word):

		word = word.lower()

		if word in self._id2word:
			return self._word2id[word]
		else:
			return self._word2id['<unk>']


	def id2word(self, ind):

		if ind >= self._size or ind < 0:
			return '<unk>'
		else:
			return self._id2word[ind]


	def save_vocab(self, path):

		f = open(path, 'wb')
		pickle.dump(self, f)
		f.close()

		print("saved vocabulary to " + path)


	def encode_sents(self, sents, length=None, pad_token=False):

		seqs = []

		for sent in sents:
			sent = self._tokenize(sent.lower())

			if pad_token:
				encoded = [0] * (len(sent) + 2)
				encoded[0] = self._word2id['<sos>']
				encoded[-1] = self._word2id['<eos>']
				pos = 1
			else:
				encoded = [0] * len(sent)
				pos = 0

			for word in sent:
				encoded[pos] = self.word2id(word)
				pos += 1

			if length:
				if length <= len(encoded):
					encoded = encoded[:length]
					# encoded[-1] = 0
				else:
					appended = [0] * (length - len(encoded))
					encoded += appended

			seqs.append(encoded)

		return seqs


	def decode_sents(self, seqs):

		sents = []

		stop = False

		for seq in seqs:
			sent = ''
			for ind in seq[:-1]:
				word = self.id2word(ind)
				
				if word == '<eos>':
					stop = True
					sent += word
					break
				elif word == '<unk>' or word == '<pad>':
					pass
				else:
					sent += word + ' '

			if not stop:
				word = self.id2word(seq[-1])
				if word != '<unk>' and word != '<pad>':
					sent += word
				else:
					sent = sent[:-1]

			sents.append(sent)

		return sents


