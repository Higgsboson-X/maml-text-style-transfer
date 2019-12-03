import os
import torch
import copy
import datetime as dt

import numpy as np

# ----------------
import models.vae
import utils.nn
import utils.data_processor

from config.model_config import default_maml_mconf

class MAMLAdvAutoencoder:

	def __init__(self, device, num_tasks=2, init_embedding=None, mconf=default_maml_mconf):

		self.num_tasks = num_tasks
		self.mconf = mconf
		self.m = models.vae.AdvAutoencoder(device, init_embedding, mconf)

		self.device = device


	def train_maml(self, support_input_sequences, support_lengths, support_labels, support_bow_representations, support_batch_size, query_input_sequences, query_lengths, query_labels, query_bow_representations, query_batch_size, epochs, init_epoch=0):

		support_batch_generator, num_batches = utils.data_processor.get_maml_batch_generator(
			support_input_sequences, support_lengths, support_labels, 
			support_bow_representations, support_batch_size, self.num_tasks, device=self.device
		)
		query_batch_generator, _ = utils.data_processor.get_maml_batch_generator(
			query_input_sequences, query_lengths, query_labels,
			query_bow_representations, query_batch_size, self.num_tasks, device=self.device
		)

		gamma = self.mconf.gamma_init
		
		meta_autoencoder_optimizer = torch.optim.Adam(
			params=self.m.autoencoder_params, lr=self.mconf.meta_autoencoder_lr,
			betas=self.mconf.betas
		)
		meta_adversary_optimizer = torch.optim.RMSprop(
			params=self.m.adversary_params, lr=self.mconf.meta_adversarial_lr
		)
		meta_style_overall_optimizer = torch.optim.RMSprop(
			params=self.m.style_overall_params, lr=self.mconf.meta_style_overall_lr
		)

		sub_autoencoder_optimizer = torch.optim.Adam(
			params=self.m.autoencoder_params, lr=self.mconf.sub_autoencoder_lr,
			betas=self.mconf.betas
		)
		sub_adversary_optimizer = torch.optim.RMSprop(
			params=self.m.adversary_params, lr=self.mconf.sub_adversarial_lr
		)
		sub_style_overall_optimizer = torch.optim.RMSprop(
			params=self.m.style_overall_params, lr=self.mconf.sub_style_overall_lr
		)

		iterations = 0
		style_kl_weight, content_kl_weight = 0., 0.
		for epoch in range(init_epoch, epochs):
			autoencoder_epoch_loss = 0.
			adversarial_epoch_loss = 0.
			style_overall_epoch_loss = 0.

			init_state = copy.deepcopy(self.m.state_dict())

			for b in range(num_batches):
				# debug
				# if b == 1:
				# 	break

				iterations += 1

				support_batch = support_batch_generator.__next__()
				query_batch = query_batch_generator.__next__()

				if iterations < self.mconf.kl_anneal_iterations:
					style_kl_weight = self.m.get_annealed_weight(iterations, self.mconf.style_kl_weight)
					content_kl_weight = self.m.get_annealed_weight(iterations, self.mconf.content_kl_weight)

				support_autoencoder_loss, support_adv_style_loss, support_adv_content_loss, support_style_overall_loss = [], [], [], []
				query_autoencoder_loss, query_adv_style_loss, query_adv_content_loss, query_style_overall_loss = [], [], [], []

				for t in range(self.num_tasks):
					batch_task = [support_batch[i][t] for i in range(len(support_batch))]
					self.m.load_state_dict(init_state)
					sub_autoencoder_optimizer.zero_grad()
					sub_adversary_optimizer.zero_grad()
					sub_style_overall_optimizer.zero_grad()

					init_state = copy.deepcopy(self.m.state_dict())

					for step in range(self.mconf.num_updates):
						[
							reconstruction_loss,
							style_kl_loss, content_kl_loss,
							style_multitask_loss, content_multitask_loss,
							style_adversarial_entropy, content_adversarial_entropy,
							style_adversarial_loss, content_adversarial_loss,
							style_overall_pred_loss
						] = self.m._feed_batch(
							batch_task[0], batch_task[1], batch_task[2], batch_task[3],
							gamma=gamma, style_kl_weight=style_kl_weight, content_kl_weight=content_kl_weight
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

						sub_autoencoder_optimizer.zero_grad()
						sub_adversary_optimizer.zero_grad()
						sub_style_overall_optimizer.zero_grad()

						# do not propagate loss to unrelated parameters
						for param in self.m.adversary_params + self.m.style_overall_params:
							param.requires_grad = False
						for param in self.m.autoencoder_params:
							param.requires_grad = True
						composite_loss.backward(retain_graph=True)

						for param in self.m.autoencoder_params + self.m.style_overall_params:
							param.requires_grad = False
						for param in self.m.adversary_params:
							param.requires_grad = True
						style_adversarial_loss.backward(retain_graph=True)
						content_adversarial_loss.backward(retain_graph=True)
						
						for param in self.m.autoencoder_params + self.m.adversary_params:
							param.requires_grad = False
						for param in self.m.style_overall_params:
							param.requires_grad = True
						style_overall_pred_loss.backward(retain_graph=True)

						# reset to default
						for param in self.m.autoencoder_params + self.m.adversary_params + self.m.style_overall_params:
							param.requires_grad = True

						torch.nn.utils.clip_grad_norm_(self.m.parameters(), self.mconf.grad_clip)

						sub_autoencoder_optimizer.step()
						sub_adversary_optimizer.step()
						sub_style_overall_optimizer.step()

						if step == 0:
							support_autoencoder_loss.append(composite_loss)
							support_adv_style_loss.append(style_adversarial_loss)
							support_adv_content_loss.append(content_adversarial_loss)
							support_style_overall_loss.append(style_overall_pred_loss)

					# query loss
					batch_task = [query_batch[i][t] for i in range(len(query_batch))]
					[
						reconstruction_loss,
						style_kl_loss, content_kl_loss,
						style_multitask_loss, content_multitask_loss,
						style_adversarial_entropy, content_adversarial_entropy,
						style_adversarial_loss, content_adversarial_loss,
						style_overall_pred_loss
					] = self.m._feed_batch(
						batch_task[0], batch_task[1], batch_task[2], batch_task[3],
						gamma=gamma, style_kl_weight=style_kl_weight, content_kl_weight=content_kl_weight
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

					query_autoencoder_loss.append(composite_loss)
					query_adv_style_loss.append(style_adversarial_loss)
					query_adv_content_loss.append(content_adversarial_loss)
					query_style_overall_loss.append(style_overall_pred_loss)

				self.m.load_state_dict(init_state)

				meta_autoencoder_loss = torch.stack(query_autoencoder_loss).sum(0) / self.num_tasks
				meta_adv_style_loss = torch.stack(query_adv_style_loss).sum(0) / self.num_tasks
				meta_adv_content_loss = torch.stack(query_adv_content_loss).sum(0) / self.num_tasks
				meta_style_overall_loss = torch.stack(query_style_overall_loss).sum(0) / self.num_tasks

				meta_autoencoder_optimizer.zero_grad()
				meta_adversary_optimizer.zero_grad()
				meta_style_overall_optimizer.zero_grad()

				# do not propagate loss to unrelated parameters
				for param in self.m.adversary_params + self.m.style_overall_params:
					param.requires_grad = False
				for param in self.m.autoencoder_params:
					param.requires_grad = True
				meta_autoencoder_loss.backward(retain_graph=True)

				for param in self.m.autoencoder_params + self.m.style_overall_params:
					param.requires_grad = False
				for param in self.m.adversary_params:
					param.requires_grad = True
				meta_adv_style_loss.backward(retain_graph=True)
				meta_adv_content_loss.backward(retain_graph=True)

				for param in self.m.autoencoder_params + self.m.adversary_params:
					param.requires_grad = False
				for param in self.m.style_overall_params:
					param.requires_grad = True
				meta_style_overall_loss.backward(retain_graph=True)

				# reset to default
				for param in self.m.autoencoder_params + self.m.adversary_params + self.m.style_overall_params:
					param.requires_grad = True

				torch.nn.utils.clip_grad_norm_(self.m.parameters(), self.mconf.grad_clip)

				meta_autoencoder_optimizer.step()
				meta_adversary_optimizer.step()
				meta_style_overall_optimizer.step()

				init_state = copy.deepcopy(self.m.state_dict())

				autoencoder_epoch_loss += meta_autoencoder_loss.item()
				adversarial_epoch_loss += (meta_adv_style_loss + meta_adv_content_loss).item()
				style_overall_epoch_loss += meta_style_overall_loss.item()

				timestamp = dt.datetime.now().isoformat()
				msg = "[{}]: batch {}/{}, epoch {}/{}, meta_vae {:g}, meta_adv_s {:g}, meta_adv_c {:g}, meta_overall_s {:g}".format(
					timestamp, b + 1, num_batches, epoch + 1, epochs,
					meta_autoencoder_loss.item(), meta_adv_style_loss.item(), meta_adv_content_loss.item(), meta_style_overall_loss.item()
				)
				for t in range(self.num_tasks):
					msg += "\n\t(task-{}) support_vae {:g}, support_adv_s {:g}, support_adv_c {:g}, support_overall_s {:g}".format(
						t + 1, support_autoencoder_loss[t].item(), support_adv_style_loss[t].item(), 
						support_adv_content_loss[t].item(), support_style_overall_loss[t].item()
					)
					msg += "\n\t(task-{}) query_vae {:g}, query_adv_s {:g}, query_adv_c {:g}, query_overall_s {:g}".format(
						t + 1, query_autoencoder_loss[t].item(), query_adv_style_loss[t].item(), 
						query_adv_content_loss[t].item(), query_style_overall_loss[t].item()
					)
				print(msg)

			gamma = max(self.mconf.gamma_min, gamma * self.mconf.gamma_decay)

			print("-------")
			print("epoch {}/{}: acc_vae_loss {:g}, acc_adv_loss {:g}, acc_style_ovrl {:g}".format(
				epoch + 1, epochs, autoencoder_epoch_loss, adversarial_epoch_loss, style_overall_epoch_loss
			))
			print("\tgamma = {:g}, style_kl_weight = {:g}, content_kl_weight = {:g}".format(
				gamma, style_kl_weight, content_kl_weight
			))


	def fine_tune(self, input_sequences_all, lengths_all, labels_all, bow_representations_all, batch_size, epochs, init_epoch=0):

		self.m.train(input_sequences_all, lengths_all, labels_all, bow_representations_all, batch_size, epochs, init_epoch)


	def get_batch_style_embeddings(self, input_sequences, lengths):

		return self.m.get_batch_style_embeddings(input_sequences, lengths)


	def infer(self, input_sequences, lengths, style_conditioning_embedding):

		return self.m.infer(input_sequences, lengths, style_conditioning_embedding)


	def save_model(self, path):

		self.m.save_model(path)


	def load_model(self, path):

		self.m.load_model(path)


