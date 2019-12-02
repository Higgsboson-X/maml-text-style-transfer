import numpy as np
import torch

def gumbel_softmax(logits, gamma, eps=1e-20):

	U = torch.randn(logits.shape)
	G = -torch.log(-torch.log(U + eps) + eps)

	return torch.nn.functional.softmax((logits + G) / gamma)


def softsample_word(dropout, proj, embedding, gamma):

	def loop_func(output):

		output = torch.nn.functional.dropout(output, dropout)
		logits = proj(output)
		prob = torch.nn.functional.gumbel_softmax(logits, gamma)

		x = torch.mm(prob, embedding)

		return x, logits

	return loop_func


def softmax_word(dropout, proj, embedding, gamma):

	def loop_func(output):

		output = torch.nn.functional.dropout(output, dropout)
		logits = proj(output)
		prob = torch.nn.functional.softmax(logits/gamma)
		x = torch.mm(prob, embedding)

		return x, logits

	return loop_func


def argmax_word(dropout, proj, embedding):

	def loop_func(output):

		output = torch.nn.functional.dropout(output, dropout)
		logits = proj(output)
		word = torch.argmax(logits, axis=1)
		x = embedding(word)

		return x, logits

	return loop_func


def rnn_decode(h, x, length, cell, loop_func):

	h_seq, logits_seq = [], []

	for t in range(length):
		h_seq.append(h.view(-1, 1, cell.hidden_size))
		output, h = cell(x.unsqueeze(1), h)
		x, logits = loop_func(output.view(-1, cell.hidden_size))
		logits_seq.append(logits.unsqueeze(1))

	return torch.cat(h_seq, axis=1), torch.cat(logits_seq, axis=1)