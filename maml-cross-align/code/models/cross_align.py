import os
import torch
import datetime as dt

import numpy as np

# ----------------
import models.discriminator
import utils.nn
import utils.data_processor

from config.model_config import default_mconf

class CrossAlign(torch.nn.Module):

	def __init__(self, device, init_embedding=None, mconf=default_mconf):

		super(CrossAlign, self).__init__()

		self.mconf = mconf
		self.device = device

		# =======================================================================
		# weights
		# embedding
		if init_embedding is None:
			self.embedding = torch.nn.Embedding(self.mconf.vocab_size, self.mconf.embedding_size)
		else:
			self.embedding = torch.nn.Embedding.from_pretrained(
				torch.tensor(init_embedding, dtype=torch.float32), 
				freeze=False
			)

		# projection to vocab
		self.proj = torch.nn.Linear(self.mconf.dim_h, self.mconf.vocab_size)
		# encode labels, encoder, onehot
		self.enc_label2y = torch.nn.Linear(self.mconf.num_labels, self.mconf.dim_y)
		# encoder
		self.enc_rnn = torch.nn.GRU(
			self.mconf.embedding_size, self.mconf.dim_h,
			batch_first=True
		)
		# self.encoder = models.encoder.SimpleDynamicEncoder() # TBC

		# encode labels, generator
		self.dec_label2y = torch.nn.Linear(self.mconf.num_labels, self.mconf.dim_y)
		# decoder
		self.dec_rnn = torch.nn.GRU(
			self.mconf.embedding_size, self.mconf.dim_h,
			batch_first=True
		)
		# self.soft_decoder = models.decoder.SoftDecoder(self.mconf.embedding_size, self.mconf.dim_h, self.proj) # TBC
		# self.hard_decoder = models.decoder.HardDecoder(self.mconf.embedding_size, self.mconf.dim_h, self.proj) # TBC

		# discriminator
		self.discriminators = []
		for s in range(self.mconf.num_labels):
			discriminator = models.discriminator.CNN(
				device=self.device, input_dim=self.mconf.dim_h, filter_sizes=self.mconf.filter_sizes, 
				n_filters=self.mconf.n_filters, dropout=self.mconf.dropout
			) # TBC
			self.discriminators.append(discriminator)

		self.to(device=self.device)


	def forward(self, input_sequence, lengths, training=True, src=0, tgt=1, gamma=None):

		if training:
			assert gamma is not None, "gamma not provided in training mode"
			loop_func = utils.nn.softsample_word(
				dropout=self.mconf.dropout, proj=self.proj,
				embedding=self.embedding.weight, gamma=gamma
			)
		else:
			loop_func = utils.nn.argmax_word(
				dropout=0., proj=self.proj,
				embedding=self.embedding
			)

		# define inputs
		batch_size = int(input_sequence.size(0)/2)
		sos = torch.empty((2*batch_size, 1), dtype=torch.int32, device=self.device).fill_(self.mconf.sos_id)
		# input sequences
		# encoder_input_sequence = torch.flip(input_sequence, dims=(1,))
		encoder_input_sequence = input_sequence
		decoder_input_sequence = torch.cat((
				sos, input_sequence
			), axis=1
		)

		# embedded input
		embedded_encoder_inputs = self.embedding(torch.as_tensor(encoder_input_sequence, dtype=torch.long, device=self.device))
		embedded_decoder_inputs = self.embedding(torch.as_tensor(decoder_input_sequence, dtype=torch.long, device=self.device))

		# targets
		'''
		targets = input_sequence.clone()
		targets[range(batch_size), lengths] = self.mconf.eos_id
		'''

		# initial state
		src_labels = torch.as_tensor(
			torch.nn.functional.one_hot(
				torch.empty((batch_size), dtype=torch.long, device=self.device).fill_(src), 
				num_classes=self.mconf.num_labels
			), dtype=torch.float32, device=self.device
		)
		tgt_labels = torch.as_tensor(
			torch.nn.functional.one_hot(
				torch.empty((batch_size), dtype=torch.long, device=self.device).fill_(tgt), 
				num_classes=self.mconf.num_labels
			), dtype=torch.float32, device=self.device
		)
		labels_ori = torch.cat((src_labels, tgt_labels), axis=0)
		labels_tsf = torch.cat((tgt_labels, src_labels), axis=0)
		h0 = torch.cat((
			self.enc_label2y(labels_ori), torch.zeros((input_sequence.size(0), self.mconf.dim_z), device=self.device)
			), axis=1
		)
		# encode embedded input sequence, get last hidden layer
		embedded_encoder_inputs = torch.nn.utils.rnn.pack_padded_sequence(
			embedded_encoder_inputs, lengths=lengths, 
			batch_first=True, enforce_sorted=False
		)
		_, z = self.enc_rnn(embedded_encoder_inputs, h0.unsqueeze(0))
		# single layer, single direction
		y = z.view(-1, self.mconf.dim_h)[:, :self.mconf.dim_y]
		z = z.view(-1, self.mconf.dim_h)[:, self.mconf.dim_y:]

		h_ori = torch.cat((
				self.dec_label2y(labels_ori), z
			), axis=1
		)
		h_tsf = torch.cat((
				self.dec_label2y(labels_tsf), z
			), axis=1
		)

		# dynamic
		packed_embedded_decoder_inputs = torch.nn.utils.rnn.pack_padded_sequence(
			embedded_decoder_inputs, lengths=lengths + 1,
			batch_first=True, enforce_sorted=False
		)
		g_outputs, _ = self.dec_rnn(packed_embedded_decoder_inputs, h_ori.unsqueeze(0))
		# second output contains the lengths of the sequences
		padded_g_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(g_outputs, batch_first=True)

		teach_h = torch.cat((
				torch.unsqueeze(h_ori, 1), padded_g_outputs
			), axis=1
		)

		g_outputs = torch.dropout(g_outputs.data, self.mconf.dropout, training)
		g_outputs = g_outputs.view(-1, self.mconf.dim_h)

		g_logits = self.proj(g_outputs)

		# reconstruction loss
		'''
		loss = torch.nn.CrossEntropyLoss()
		targets = torch.nn.utils.rnn.pack_padded_sequence(
			targets, lengths=lengths+1,
			batch_first=True, enforce_sorted=False
		).data
		rec_loss = loss(targets.view(-1), g_logits, reduction="none")
		rec_loss = rec_loss / float(batch_size)
		'''

		# decode
		go = embedded_decoder_inputs[:, 0, :]
		dec_h_tsf, dec_logits_tsf = utils.nn.rnn_decode(
			h=h_tsf.unsqueeze(0), x=go, length=self.mconf.max_seq_length,
			cell=self.dec_rnn, loop_func=loop_func
		)

		return teach_h, g_logits, dec_h_tsf, dec_logits_tsf, y, z


	def _feed_batch(self, input_sequence, lengths, gamma):

		# input_sequence contains same number of sequences from style [0] to style [num_labels-1]

		tot_batch_size = input_sequence.size(0)
		sub_batch_size = int(tot_batch_size/self.mconf.num_labels)

		targets = torch.cat((
				input_sequence.clone(), torch.empty((tot_batch_size, 1), dtype=torch.int32, device=self.device).fill_(self.mconf.pad_id)
			), axis=-1
		)
		targets[range(tot_batch_size), lengths.cpu().numpy()] = self.mconf.eos_id

		loss_rec = torch.tensor(0., device=self.device)
		loss_adv = torch.tensor(0., device=self.device)

		loss_ds = torch.zeros(self.mconf.num_labels, dtype=torch.float32, device=self.device)

		loss_rec_fn = torch.nn.CrossEntropyLoss(reduction="none")

		for i in range(self.mconf.num_labels):
			for j in range(i + 1,self.mconf.num_labels):
				start_src = sub_batch_size * i
				end_src = sub_batch_size * (i + 1)
				start_tgt = sub_batch_size * j
				end_tgt = sub_batch_size * (j + 1)

				sub_input_sequence = torch.cat(
					(input_sequence[start_src:end_src], input_sequence[start_tgt:end_tgt])
				)
				sub_lengths = torch.cat(
					(lengths[start_src:end_src], lengths[start_tgt:end_tgt])
				)

				teach_h, g_logits, dec_h_tsf, _, _, _ = self.forward(
					sub_input_sequence, lengths=sub_lengths, 
					training=True, src=i, tgt=j, gamma=gamma
				)

				loss_d0, loss_g0 = models.discriminator.discriminate(
					device=self.device, 
					x_real=teach_h[:sub_batch_size], x_fake=dec_h_tsf[sub_batch_size:],
					cnn=self.discriminators[i], eta=self.mconf.eta
				)
				loss_d1, loss_g1 = models.discriminator.discriminate(
					device=self.device,
					x_real=teach_h[sub_batch_size:], x_fake=dec_h_tsf[:sub_batch_size],
					cnn=self.discriminators[j], eta=self.mconf.eta
				)

				targets_merged = torch.nn.utils.rnn.pack_padded_sequence(
					torch.cat((targets[start_src:end_src], targets[start_tgt:end_tgt])),
					lengths=sub_lengths + 1, batch_first=True, enforce_sorted=False
				).data.view(-1)

				loss = loss_rec_fn(
					target=torch.as_tensor(targets_merged, dtype=torch.long, device=self.device), input=g_logits
				)
				loss_rec += torch.sum(loss) / float(2*sub_batch_size)
				loss_adv += loss_g0 + loss_g1

				loss_ds[i] += loss_d0
				loss_ds[j] += loss_d1

		return loss_rec, loss_adv, loss_ds


	def train(self, input_sequence_all, lengths_all, batch_size, epochs, init_epoch=0, pretrain=False):

		# input_sequence_all is a list of sequences of the same size for all styles
		batch_generator, num_batches = utils.data_processor.get_batch_generator(
			input_sequence_all, lengths_all, 
			batch_size, device=self.device
		)
		gamma = self.mconf.gamma_init
		optimizer = torch.optim.Adam(
			self.parameters(), lr=self.mconf.train_lr,
			betas=self.mconf.betas
		)
		
		for epoch in range(init_epoch, epochs):
			epoch_loss = 0.
			for b in range(num_batches):
				# debug
				# if b == 1:
				# 	break
				batch = batch_generator.__next__()

				loss_rec, loss_adv, loss_ds = self._feed_batch(
					batch[0], batch[1], gamma=gamma
				)
				if torch.sum(loss_ds <= self.mconf.d_loss_tolerance) == self.mconf.num_labels:
					loss = loss_rec + self.mconf.adv_loss_weight * loss_adv + sum(loss_ds)
				elif pretrain:
					# only optimizer reconstruction loss
					loss = loss_rec
				else:
					loss = loss_rec + sum(loss_ds)

				# optimize
				optimizer.zero_grad()
				loss.backward()
				torch.nn.utils.clip_grad_norm_(self.parameters(), self.mconf.grad_clip)
				optimizer.step()

				epoch_loss += loss.item()

				timestamp = dt.datetime.now().isoformat()
				msg = "[{}]: batch {}/{}, epoch {}/{}, loss {:g}, loss_rec {:g}, loss_adv {:g}, ".format(
					timestamp, b + 1, num_batches, epoch + 1, epochs,
					loss.item(), loss_rec.item(), loss_adv.item()
				)
				for l in range(self.mconf.num_labels):
					msg += " loss_d{} {:g}".format(l, loss_ds[l].item())
				print(msg)

			gamma = max(self.mconf.gamma_min, gamma * self.mconf.gamma_decay)

			print("-------")
			print("epoch {}/{}: accumulated_loss {:g}".format(epoch + 1, epochs, epoch_loss))



	def infer(self, input_sequence, lengths, src, tgt):

		# src: style id for source
		# tgt: style id for target
		ori_size = lengths.shape[0]
		input_sequence = torch.cat(
			[torch.tensor(input_sequence, dtype=torch.int32, device=self.device)] * 2, 
			axis=0
		)
		lengths = torch.cat(
			[torch.tensor(lengths, dtype=torch.int32, device=self.device)] * 2, 
			axis=0
		)
		_, _, _, dec_logits_tsf, _, _ = self.forward(
			input_sequence, 
			lengths=lengths, 
			training=False, src=src, tgt=tgt
		)
		sampled_seq = torch.argmax(dec_logits_tsf[:ori_size], axis=-1)

		return sampled_seq


	def get_batch_embeddings(self, input_sequence, lengths):

		input_sequence = torch.tensor(input_sequence, dtype=torch.int32, device=self.device)
		lengths = torch.tensor(lengths, dtype=torch.int32, device=self.device)
		# input sequence contains half of original sequence and other half the target sequence
		[
			_, _, 
			_, _, y, z
		] = self.forward(input_sequence, lengths, training=False)

		# y = style embedding, z = content embedding

		return y, z


	def save_model(self, path):

		torch.save(self.state_dict(), path)
		print("saved model to {}".format(path))


	def load_model(self, path):

		self.load_state_dict(torch.load(path, map_location=self.device))
		print("loaded model from {}".format(path))



