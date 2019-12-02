import os
import torch
import datetime as dt

import numpy as np

# ----------------

import utils.nn
from config.model_config import default_mconf

class AdvAutoencoder(torch.nn.Module):

	def __init__(self, device, init_embedding=None, mconf=default_mconf):

		super(AdvAutoencoder, self).__init__()

		self.mconf = mconf
		self.device = device

		# print(self.mconf.__dict__)

		# =======================================================================
		# weights
		# embedding
		if init_embedding is None:
			self.encoder_embedding = torch.nn.Embedding(self.mconf.vocab_size, self.mconf.embedding_size)
			self.decoder_embedding = torch.nn.Embedding(self.mconf.vocab_size, self.mconf.embedding_size)
		else:
			self.encoder_embedding = torch.nn.Embedding.from_pretrained(init_embedding, freeze=False)
			self.decoder_embedding = torch.nn.Embedding.from_pretrained(init_embedding, freeze=False)

		# encoder
		self.encoder = torch.nn.GRU(
			input_size=self.mconf.embedding_size, 
			hidden_size=self.mconf.encoder_rnn_size,
			batch_first=True, bidirectional=True
		)
		# sentence_embedding_size = 2 * encoder_rnn_size (bidirectional)
		# style embedding
		self.style_embedding_mu_fc = torch.nn.Linear(self.mconf.sentence_embedding_size, self.mconf.style_embedding_size)
		self.style_embedding_sigma_fc = torch.nn.Linear(self.mconf.sentence_embedding_size, self.mconf.style_embedding_size)
		
		# content embedding
		self.content_embedding_mu_fc = torch.nn.Linear(self.mconf.sentence_embedding_size, self.mconf.content_embedding_size)
		self.content_embedding_sigma_fc = torch.nn.Linear(self.mconf.sentence_embedding_size, self.mconf.content_embedding_size)
		
		# generative embedding layer
		self.generative_embedding_fc = torch.nn.Linear(
			in_features=self.mconf.style_embedding_size + self.mconf.content_embedding_size,
			out_features=self.mconf.decoder_rnn_size
		)
		
		# decoder
		# input = generative_embedding : embedded_decoder_input
		self.decoder = torch.nn.GRU(
			input_size=self.mconf.embedding_size + self.mconf.decoder_rnn_size,
			hidden_size=self.mconf.decoder_rnn_size,
			batch_first=True
		)
		self.decoder_proj = torch.nn.Linear(self.mconf.decoder_rnn_size, self.mconf.vocab_size, bias=False)

		# style adversarial
		self.style_adversarial_mlp_fc = torch.nn.Linear(self.mconf.content_embedding_size, self.mconf.content_embedding_size)
		self.style_adversarial_pred_fc = torch.nn.Linear(self.mconf.content_embedding_size, self.mconf.num_labels)
		# content adversarial
		self.content_adversarial_mlp_fc = torch.nn.Linear(self.mconf.style_embedding_size, self.mconf.vocab_size)
		self.content_adversarial_pred_fc = torch.nn.Linear(self.mconf.vocab_size, self.mconf.vocab_size)

		# multitask layers
		self.style_multitask_pred_fc = torch.nn.Linear(self.mconf.style_embedding_size, self.mconf.num_labels)
		self.content_multitask_pred_fc = torch.nn.Linear(self.mconf.content_embedding_size, self.mconf.vocab_size)

		# overall latent space classifier
		# used to prove disentanglement
		self.style_overall_pred_fc = torch.nn.Linear(
			in_features=self.mconf.style_embedding_size + self.mconf.content_embedding_size,
			out_features=self.mconf.num_labels
		)

		# params
		self.autoencoder_params = list(self.encoder_embedding.parameters()) + list(self.decoder_embedding.parameters()) \
					+ list(self.encoder.parameters()) + list(self.decoder.parameters()) \
					+ list(self.style_embedding_mu_fc.parameters()) + list(self.style_embedding_sigma_fc.parameters()) \
					+ list(self.content_embedding_mu_fc.parameters()) + list(self.content_embedding_sigma_fc.parameters()) \
					+ list(self.style_multitask_pred_fc.parameters()) + list(self.content_multitask_pred_fc.parameters()) \
					+ list(self.generative_embedding_fc.parameters()) + list(self.decoder_proj.parameters())
		self.adversary_params = list(self.style_adversarial_mlp_fc.parameters()) + list(self.style_adversarial_pred_fc.parameters()) \
					+ list(self.content_adversarial_mlp_fc.parameters()) + list(self.content_adversarial_pred_fc.parameters())

		self.style_overall_params = list(self.style_overall_pred_fc.parameters())

		self.to(device=self.device)


	def forward(self, input_sequences, lengths, training=True, style_conditioning_embeddings=None, gamma=0.01):

		if training:
			assert gamma is not None, "gamma not provided in training mode"
			loop_func = utils.nn.softsample_word(
				dropout=self.mconf.rnn_dropout, proj=self.decoder_proj,
				embedding=self.decoder_embedding.weight, gamma=gamma
			)
		else:
			loop_func = utils.nn.argmax_word(
				dropout=0., proj=self.decoder_proj,
				embedding=self.decoder_embedding
			)
			assert style_conditioning_embeddings is not None, "style_conditioning_embeddings not provided"

		batch_size = int(input_sequences.size(0))
		sos = torch.empty((batch_size, 1), dtype=torch.int32, device=self.device).fill_(self.mconf.sos_id)
		# input sequences
		encoder_input_sequences = input_sequences
		decoder_input_sequences = torch.cat((
				sos, input_sequences
			), axis=1
		)

		# embedded
		embedded_encoder_inputs = self.encoder_embedding(torch.as_tensor(encoder_input_sequences, dtype=torch.long, device=self.device))
		embedded_decoder_inputs = self.decoder_embedding(torch.as_tensor(decoder_input_sequences, dtype=torch.long, device=self.device))
		# dropout
		embedded_encoder_inputs = torch.dropout(embedded_encoder_inputs, self.mconf.sequence_word_dropout, training)
		embedded_decoder_inputs = torch.dropout(embedded_decoder_inputs, self.mconf.sequence_word_dropout, training)

		# sentence embedding
		packed_embedded_encoder_inputs = torch.nn.utils.rnn.pack_padded_sequence(
			embedded_encoder_inputs, lengths=lengths,
			batch_first=True, enforce_sorted=False
		)
		h0 = torch.zeros(batch_size, self.mconf.encoder_rnn_size, device=self.device)
		_, sentence_embeddings = self.encoder(
			packed_embedded_encoder_inputs, torch.cat(
				[h0.unsqueeze(0)] * 2, axis=0
			)
		)
		sentence_embeddings = torch.dropout(torch.cat(
				(sentence_embeddings[0], sentence_embeddings[1]), axis=1
			), p=self.mconf.rnn_dropout, train=training
		)

		# style embeddings
		style_embeddings_mu = torch.dropout(
			torch.nn.functional.leaky_relu(self.style_embedding_mu_fc(sentence_embeddings)),
			p=self.mconf.fc_dropout, train=training
		)
		style_embeddings_sigma = torch.dropout(
			torch.nn.functional.leaky_relu(self.style_embedding_sigma_fc(sentence_embeddings)), 
			p=self.mconf.fc_dropout, train=training
		)

		sampled_style_embeddings = self.sample_prior(style_embeddings_mu, style_embeddings_sigma)

		'''
		print(sampled_style_embeddings.shape)
		if style_conditioning_embeddings is not None:
			print(style_conditioning_embeddings.shape)
		'''

		style_embeddings = sampled_style_embeddings if training else style_conditioning_embeddings

		# content embeddings
		content_embeddings_mu = torch.dropout(
			torch.nn.functional.leaky_relu(self.content_embedding_mu_fc(sentence_embeddings)),
			p=self.mconf.fc_dropout, train=training
		)
		content_embeddings_sigma = torch.dropout(
			torch.nn.functional.leaky_relu(self.content_embedding_mu_fc(sentence_embeddings)),
			p=self.mconf.fc_dropout, train=training
		)

		sampled_content_embeddings = self.sample_prior(content_embeddings_mu, content_embeddings_sigma)

		content_embeddings = sampled_content_embeddings if training else content_embeddings_mu

		# generative_embeddings from concatenated style_embeddings : content_embeddings
		generative_embeddings = torch.nn.functional.leaky_relu(
			self.generative_embedding_fc(torch.cat((
					style_embeddings, content_embeddings
				), axis=1
			))
		)

		# decoded sequence
		concatenated_decoder_input = torch.cat((
				torch.cat([generative_embeddings.unsqueeze(1)] * embedded_decoder_inputs.size(1), axis=1), 
				embedded_decoder_inputs
			), axis=-1
		)
		# print(concatenated_decoder_input.shape)
		packed_embedded_decoder_inputs = torch.nn.utils.rnn.pack_padded_sequence(
			concatenated_decoder_input, lengths=lengths, batch_first=True, enforce_sorted=False
		)
		h0 = torch.zeros(batch_size, self.mconf.decoder_rnn_size, device=self.device)
		g_outputs, _ = self.decoder(packed_embedded_decoder_inputs, h0.unsqueeze(0))
		# print(g_outputs.data.shape)
		# padded_g_outputs, _ torch.nn.utils.rnn.pad_packed_sequence(g_outputs, batch_first=True)
		g_outputs = torch.dropout(g_outputs.data, self.mconf.rnn_dropout, train=training)
		g_logits = self.decoder_proj(g_outputs.view(-1, self.mconf.decoder_rnn_size))

		# sos = torch.empty((batch_size, 1), dtype=torch.int32, device=self.device).fill_(self.mconf.sos_id)
		go = embedded_encoder_inputs[:, 0, :]
		# h0 = torch.zeros(batch_size, self.mconf.decoder_rnn_size, device=self.device)
		_, dec_logits_tsf = utils.nn.rnn_decode_with_latent_vec(
			h=h0.unsqueeze(0), x=go, length=self.mconf.max_seq_length,
			cell=self.decoder, loop_func=loop_func, latent_vec=generative_embeddings
		)

		# overall prediction
		style_overall_pred = torch.dropout(torch.nn.functional.softmax(
			self.style_overall_pred_fc(torch.cat(
				(style_embeddings_mu, content_embeddings_mu), axis=1
			)), dim=1), p=self.mconf.fc_dropout, train=training
		)

		results = [
			style_embeddings_mu, style_embeddings_sigma, style_embeddings,
			content_embeddings_mu, content_embeddings_sigma, content_embeddings,
			generative_embeddings, g_logits, dec_logits_tsf, style_overall_pred
		]

		return results

	def _feed_batch(self, input_sequences, lengths, labels, bow_representations, gamma, style_kl_weight, content_kl_weight):

		# specifically for training
		batch_size = input_sequences.size(0)
		targets = torch.cat((
				input_sequences.clone(), torch.empty((batch_size, 1), dtype=torch.int32, device=self.device).fill_(self.mconf.pad_id)
			), axis=-1
		)
		targets[range(batch_size), lengths.cpu().numpy()] = self.mconf.eos_id
		# embedded_decoder_inputs = self.decoder_embedding(decoder_input_sequences)
		# embedded_decoder_inputs = torch.dropout(embedded_decoder_inputs, p=self.mconf.sequence_word_dropout, train=True)


		[
			style_embeddings_mu, style_embeddings_sigma, style_embeddings,
			content_embeddings_mu, content_embeddings_sigma, content_embeddings,
			generative_embeddings, g_logits, dec_logits_tsf, style_overall_pred
		] = self.forward(input_sequences, lengths, training=True, gamma=gamma)

		# style adversary
		style_adversarial_mlp = torch.dropout(
			torch.nn.functional.leaky_relu(self.style_adversarial_mlp_fc(content_embeddings_mu)),
			p=self.mconf.fc_dropout, train=True
		)
		style_adversarial_pred = torch.nn.functional.softmax(self.style_adversarial_pred_fc(style_adversarial_mlp), dim=1)

		style_adversarial_entropy = utils.nn.compute_batch_entropy(style_adversarial_pred, eps=self.mconf.eps)
		style_adversarial_loss = utils.nn.reduced_cross_entropy_loss(
			onehot=torch.nn.functional.one_hot(torch.as_tensor(labels, dtype=torch.long, device=self.device), num_classes=self.mconf.num_labels), 
			pred=style_adversarial_pred, eps=self.mconf.eps
		)

		# content adversary
		content_adversarial_mlp = torch.dropout(
			torch.nn.functional.leaky_relu(self.content_adversarial_mlp_fc(style_embeddings)),
			p=self.mconf.fc_dropout, train=True
		)
		content_adversarial_pred = torch.nn.functional.softmax(self.content_adversarial_pred_fc(content_adversarial_mlp), dim=1)

		content_adversarial_entropy = utils.nn.compute_batch_entropy(content_adversarial_pred, eps=self.mconf.eps)
		content_adversarial_loss = utils.nn.reduced_cross_entropy_loss(
			onehot=bow_representations, pred=content_adversarial_pred, eps=self.mconf.eps
		)

		# multitask loss
		style_multitask_pred = torch.dropout(
			torch.nn.functional.softmax(self.style_multitask_pred_fc(style_embeddings_mu), dim=1),
			p=self.mconf.fc_dropout, train=True
		)
		style_multitask_loss = utils.nn.reduced_cross_entropy_loss(
			onehot=torch.nn.functional.one_hot(torch.as_tensor(labels, dtype=torch.long, device=self.device), num_classes=self.mconf.num_labels),
			pred=style_multitask_pred, eps=self.mconf.eps
		)
		content_multitask_pred = torch.dropout(
			torch.nn.functional.softmax(self.content_multitask_pred_fc(content_embeddings_mu), dim=1),
			p=self.mconf.fc_dropout, train=True
		)
		content_multitask_loss = utils.nn.reduced_cross_entropy_loss(
			onehot=bow_representations, pred=content_multitask_pred, eps=self.mconf.eps
		)

		# kl loss
		style_kl_loss = utils.nn.calc_kl_loss(style_embeddings_mu, style_embeddings_sigma)
		content_kl_loss = utils.nn.calc_kl_loss(content_embeddings_mu, content_embeddings_sigma)

		# reconstruction loss
		seq_cel_fn = torch.nn.CrossEntropyLoss(reduction="none")
		packed_targets = torch.nn.utils.rnn.pack_padded_sequence(
			torch.as_tensor(targets, dtype=torch.long, device=self.device), lengths=lengths,
			batch_first=True, enforce_sorted=False
		).data.view(-1)
		reconstruction_loss = seq_cel_fn(
			target=packed_targets,
			input=g_logits
		)
		reconstruction_loss = torch.sum(reconstruction_loss) / float(batch_size)

		# style overall prediction loss
		style_overall_pred_loss = seq_cel_fn(
			target=torch.as_tensor(labels, dtype=torch.long, device=self.device),
			input=style_overall_pred
		)
		style_overall_pred_loss = torch.sum(style_overall_pred_loss) / float(batch_size)

		losses = [
			reconstruction_loss,
			style_kl_loss, content_kl_loss,
			style_multitask_loss, content_multitask_loss,
			style_adversarial_entropy, content_adversarial_entropy,
			style_adversarial_loss, content_adversarial_loss,
			style_overall_pred_loss
		]

		return losses


	def sample_prior(self, mu, log_sigma):

		epsilon = torch.randn(log_sigma.shape, dtype=torch.float32, device=self.device)

		return mu + epsilon * torch.exp(log_sigma)


	def get_annealed_weight(self, iterations, weight):

		return (np.tanh(
				(iterations - self.mconf.kl_anneal_iterations*1.5) / (self.mconf.kl_anneal_iterations/3)
			) + 1) * weight


	def train(self, input_sequences_all, lengths_all, labels_all, bow_representations_all, batch_size=32, epochs=5, init_epoch=0):

		batch_generator, num_batches = utils.data_processor.get_batch_generator(
			input_sequences_all, lengths_all, labels_all, 
			bow_representations_all, batch_size, device=self.device
		)
		gamma = self.mconf.gamma_init

		'''
		autoencoder_optimizer = torch.optim.Adam(
			params=[
				self.encoder_embedding.parameters(), self.decoder_embedding.parameters(),
				self.encoder.parameters(), self.decoder.parameters(),
				self.style_embedding_mu_fc.parameters(), self.style_embedding_sigma_fc.parameters(),
				self.content_embedding_mu_fc.parameters(), self.content_embedding_sigma_fc.parameters(),
				self.style_multitask_pred_fc.parameters(), self.content_multitask_pred_fc.parameters(),
				self.generative_embedding_fc.parameters(), self.decoder_proj.parameters()
			], lr=self.mconf.autoencoder_train_lr, betas=self.mconf.betas
		)
		adversary_optimizer = torch.optim.RMSprop(
			params=[
				self.style_adversarial_mlp_fc.parameters(), 
				self.style_adversarial_pred_fc.parameters(),
				self.content_adversarial_mlp_fc.parameters(),
				self.content_adversarial_pred_fc.parameters()
			], lr=self.mconf.adversarial_train_lr
		)
		'''
		autoencoder_optimizer = torch.optim.Adam(
			params=self.autoencoder_params, lr=self.mconf.autoencoder_train_lr, betas=self.mconf.betas
		)
		adversary_optimizer = torch.optim.RMSprop(
			params=self.adversary_params, lr=self.mconf.adversarial_train_lr
		)
		style_overall_optimizer = torch.optim.RMSprop(
			params=self.style_overall_params, lr=self.mconf.style_overall_train_lr
		)

		iterations = 0
		style_kl_weight, content_kl_weight = 0., 0.
		for epoch in range(init_epoch, epochs):
			autoencoder_epoch_loss = 0.
			adversarial_epoch_loss = 0.
			style_overall_epoch_loss = 0.

			for b in range(num_batches):
				# debug
				# if b == 1:
				#	break

				iterations += 1

				batch = batch_generator.__next__()

				if iterations < self.mconf.kl_anneal_iterations:
					style_kl_weight = self.get_annealed_weight(iterations, self.mconf.style_kl_weight)
					content_kl_weight = self.get_annealed_weight(iterations, self.mconf.content_kl_weight)

				[
					reconstruction_loss,
					style_kl_loss, content_kl_loss,
					style_multitask_loss, content_multitask_loss,
					style_adversarial_entropy, content_adversarial_entropy,
					style_adversarial_loss, content_adversarial_loss,
					style_overall_pred_loss
				] = self._feed_batch(
					batch[0], batch[1], batch[2], batch[3], gamma=gamma, 
					style_kl_weight=style_kl_weight, content_kl_weight=content_kl_weight
				)

				'''
				if style_adversarial_loss <= self.mconf.adv_loss_tolerance and content_adversarial_loss <= self.mconf.adv_loss_tolerance:
					composite_loss = reconstruction_loss + style_kl_weight * style_kl_loss + content_kl_weight * content_kl_loss \
							+ self.mconf.style_multitask_loss_weight * style_multitask_loss \
							+ self.mconf.content_multitask_loss_weight * content_multitask_loss \
							- self.mconf.style_adv_loss_weight * style_adversarial_entropy \
							- self.mconf.content_adv_loss_weight * content_adversarial_entropy
				else:
					composite_loss = reconstruction_loss + style_kl_weight * style_kl_loss + content_kl_weight * content_kl_loss \
							+ style_multitask_loss + content_multitask_loss
				'''
				composite_loss = reconstruction_loss + style_kl_weight * style_kl_loss + content_kl_weight * content_kl_loss \
						+ self.mconf.style_multitask_loss_weight * style_multitask_loss \
						+ self.mconf.content_multitask_loss_weight * content_multitask_loss \
						- self.mconf.style_adv_loss_weight * style_adversarial_entropy \
						- self.mconf.content_adv_loss_weight * content_adversarial_entropy

				autoencoder_optimizer.zero_grad()
				adversary_optimizer.zero_grad()
				style_overall_optimizer.zero_grad()

				composite_loss.backward(retain_graph=True)
				style_adversarial_loss.backward(retain_graph=True)
				content_adversarial_loss.backward(retain_graph=True)
				style_overall_pred_loss.backward(retain_graph=True)

				# grad clip for all parameters
				torch.nn.utils.clip_grad_norm_(self.parameters(), self.mconf.grad_clip)

				autoencoder_optimizer.step()
				adversary_optimizer.step()
				style_overall_optimizer.step()

				autoencoder_epoch_loss += composite_loss.item()
				adversarial_epoch_loss += (content_adversarial_loss + style_adversarial_loss).item()
				style_overall_epoch_loss += style_overall_pred_loss.item()

				timestamp = dt.datetime.now().isoformat()
				msg = "[{}]: batch {}/{}, epoch {}/{}, loss_vae {:g}, style_adversarial_loss {:g}, content_adversarial_loss {:g}".format(
					timestamp, b + 1, num_batches, epoch + 1, epochs, 
					composite_loss.item(), style_adversarial_loss.item(), content_adversarial_loss.item(),
				)
				msg += "\n\t reconstruction_loss {:g}, style_kl_loss {:g}, content_kl_loss {:g}".format(
					reconstruction_loss.item(), style_kl_loss.item(), content_kl_loss.item()
				)
				msg += "\n\t style_multitask_loss {:g}, content_multitask_loss {:g}, style_overall_pred_loss {:g}".format(
					style_multitask_loss.item(), content_multitask_loss.item(), style_overall_pred_loss.item()
				)
				print(msg)

			gamma = max(self.mconf.gamma_min, gamma * self.mconf.gamma_decay)

			print("-------")
			print("epoch {}/{}: autoencoder_epoch_loss {:g}, adversarial_epoch_loss {:g}, style_overall_epoch_loss {:g}".format(
				epoch + 1, epochs, autoencoder_epoch_loss, style_overall_epoch_loss, adversarial_epoch_loss
			))

	def get_batch_style_embeddings(self, input_sequences, lengths):

		input_sequences = torch.tensor(input_sequences, dtype=torch.int32, device=self.device)
		lengths = torch.tensor(lengths, dtype=torch.int32, device=self.device)

		[
			_, _, style_embeddings,
			_, _, _,
			_, _, _, _
		] = self.forward(
			input_sequences=input_sequences, lengths=lengths,
			training=True, style_conditioning_embeddings=None, gamma=self.mconf.gamma_init
		)

		return style_embeddings


	def infer(self, input_sequences, lengths, style_conditioning_embedding):

		# style_conditioning_embedding.shape == (style_embedding_size,)

		batch_size = lengths.shape[0]

		input_sequences = torch.tensor(input_sequences, dtype=torch.int32, device=self.device)
		lengths = torch.tensor(lengths, dtype=torch.int32, device=self.device)
		style_conditioning_embeddings = np.concatenate(
			[np.expand_dims(style_conditioning_embedding, axis=0)] * batch_size, axis=0
		)
		style_conditioning_embeddings = torch.tensor(style_conditioning_embeddings, dtype=torch.float32, device=self.device)

		[
			_, _, _,
			_, _, _,
			_, _, dec_logits_tsf, style_overall_pred
		] = self.forward(
			input_sequences=input_sequences, lengths=lengths, 
			training=False, style_conditioning_embeddings=style_conditioning_embeddings, gamma=None
		)

		sampled_seq = torch.argmax(dec_logits_tsf, axis=-1)
		preds = torch.argmax(style_overall_pred, axis=-1)

		return sampled_seq, preds


	def save_model(self, path):

		torch.save(self.state_dict(), path)
		print("saved model to {}".format(path))


	def load_model(self, path):

		self.load_state_dict(torch.load(path))
		print("loaded model from {}".format(path))

