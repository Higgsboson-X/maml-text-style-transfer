import os
import torch
import copy
import datetime as dt

import numpy as np

# ----------------
import models.cross_align
import models.discriminator
import utils.nn
import utils.data_processor

from config.model_config import default_maml_mconf

class MAMLCrossAlign:

	def __init__(self, device, num_tasks=2, init_embedding=None, mconf=default_maml_mconf):

		self.num_tasks = num_tasks
		self.mconf = mconf
		self.m = models.cross_align.CrossAlign(device, init_embedding, mconf)

		self.device = device


	def train_maml(self, support_input_sequence, support_lengths, support_batch_size, query_input_sequence, query_lengths, query_batch_size, epochs, init_epoch=0):

		support_batch_generator, num_batches = utils.data_processor.get_maml_batch_generator(
			support_input_sequence, support_lengths, support_batch_size, 
			self.num_tasks, device=self.device
		)
		query_batch_generator, _ = utils.data_processor.get_maml_batch_generator(
			query_input_sequence, query_lengths, query_batch_size, 
			self.num_tasks, device=self.device
		)
		
		gamma = self.mconf.gamma_init
		meta_optimizer = torch.optim.Adam(
			self.m.parameters(), lr=self.mconf.meta_lr,
			betas=self.mconf.betas
		)
		sub_optimizer = torch.optim.Adam(
			self.m.parameters(), lr=self.mconf.sub_lr,
			betas=self.mconf.betas
		)

		for epoch in range(init_epoch, epochs):
			epoch_loss = 0.
			init_state = copy.deepcopy(self.m.state_dict())
			for b in range(num_batches):
				# debug
				# if b == 1:
				# 	break
				support_batch = support_batch_generator.__next__()
				query_batch = query_batch_generator.__next__()
				support_loss = []
				query_loss = []
				for t in range(self.num_tasks):
					batch_task = [support_batch[0][t], support_batch[1][t]]
					self.m.load_state_dict(init_state)
					sub_optimizer.zero_grad()

					# init_state will change when the model state changes after loading
					init_state = copy.deepcopy(self.m.state_dict())

					# support loss
					'''
					loss_rec, loss_adv, loss_ds = self.m._feed_batch(
						batch_task[0], batch_task[1], gamma=gamma
					)
					if torch.sum(loss_ds <= self.mconf.d_loss_tolerance) == self.mconf.num_labels:
						loss = loss_rec + self.mconf.adv_loss_weight * loss_adv + sum(loss_ds)
					else:
						loss = loss_rec + sum(loss_ds)
					loss.backward()
					torch.nn.utils.clip_grad_norm_(self.m.parameters(), self.mconf.grad_clip)
					optim.step()
					support_loss.append(loss)
					'''

					for step in range(self.mconf.num_updates):
						loss_rec, loss_adv, loss_ds = self.m._feed_batch(
							batch_task[0], batch_task[1], gamma=gamma
						)
						if torch.sum(loss_ds <= self.mconf.d_loss_tolerance) == self.mconf.num_labels:
							loss = loss_rec + self.mconf.adv_loss_weight * loss_adv + sum(loss_ds)
						else:
							loss = loss_rec + sum(loss_ds)
						# support update
						sub_optimizer.zero_grad()
						loss.backward()
						torch.nn.utils.clip_grad_norm_(self.m.parameters(), self.mconf.grad_clip)
						sub_optimizer.step()
						# support loss
						if step == 0:
							support_loss.append(loss)

					# query loss
					batch_task = [query_batch[0][t], query_batch[1][t]]
					loss_rec, loss_adv, loss_ds = self.m._feed_batch(
						batch_task[0], batch_task[1], gamma=gamma
					)
					if torch.sum(loss_ds <= self.mconf.d_loss_tolerance) == self.mconf.num_labels:
						loss = loss_rec + loss_adv + sum(loss_ds)
					else:
						loss = loss_rec + sum(loss_ds)

					query_loss.append(loss)

				self.m.load_state_dict(init_state)

				loss_meta = torch.stack(query_loss).sum(0) / self.num_tasks

				meta_optimizer.zero_grad()
				loss_meta.backward()
				torch.nn.utils.clip_grad_norm_(self.m.parameters(), self.mconf.grad_clip)
				meta_optimizer.step()

				init_state = copy.deepcopy(self.m.state_dict())
				epoch_loss += loss_meta.item()

				timestamp = dt.datetime.now().isoformat()
				msg = "[{}]: batch {}/{}, epoch {}/{}, loss_meta {:g};".format(
					timestamp, b + 1, num_batches, epoch + 1, epochs,
					loss_meta.item()
				)
				for t in range(self.num_tasks):
					msg += "\n\t(task-{}) support_loss {:g}, query_loss {:g}".format(
						t + 1, support_loss[t].item(), query_loss[t].item()
					)
				print(msg)

			gamma = max(self.mconf.gamma_min, gamma * self.mconf.gamma_decay)

			print("-------")
			print("epoch {}/{} accumulated_loss {:g}".format(epoch + 1, epochs, epoch_loss))


	def fine_tune(self, input_sequence_all, lengths_all, batch_size, epochs, init_epoch=1):

		self.m.train(input_sequence_all, lengths_all, batch_size, epochs=epochs, init_epoch=init_epoch)


	def infer(self, input_sequence, lengths, src, tgt):

		return self.m.infer(input_sequence, lengths, src, tgt)


	def save_model(self, path):

		self.m.save_model(path)


	def load_model(self, path):

		self.m.load_model(path)
