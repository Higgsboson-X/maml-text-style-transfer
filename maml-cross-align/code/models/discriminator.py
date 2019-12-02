import torch

class CNN(torch.nn.Module):

	def __init__(self, device, input_dim, filter_sizes, n_filters, dropout):

		super(CNN, self).__init__()
		self.filters = []
		for size in filter_sizes:
			conv2d = torch.nn.Conv2d(
				in_channels=1, out_channels=n_filters, kernel_size=(size, input_dim),
				stride=1, padding=0
			)
			conv2d.to(device=device)
			self.filters.append(conv2d)
		self.proj = torch.nn.Linear(n_filters*len(filter_sizes), 1)
		self.dropout = dropout
		self.n_filters = n_filters
		self.filter_sizes = filter_sizes

		self.device = device

		self.to(device=device)

	def forward(self, x):

		x = x.unsqueeze(1)
		outputs = []
		for f in self.filters:
			conv = f(x)
			h = torch.nn.functional.leaky_relu(conv)
			pooled = torch.max(h, dim=-2).values.view(-1, self.n_filters)
			outputs.append(pooled)

		outputs = torch.cat(outputs, axis=1)
		outputs = torch.nn.functional.dropout(outputs, self.dropout)

		logits = self.proj(outputs).view(-1)

		return logits


def discriminate(device, x_real, x_fake, cnn, eta=10.):

	d_real = torch.sigmoid(cnn.forward(x=x_real))
	d_fake = torch.sigmoid(cnn.forward(x=x_fake))

	ones = torch.ones(d_real.shape[0], dtype=torch.float32, device=device)
	zeros = torch.zeros(d_fake.shape[0], dtype=torch.float32, device=device)

	loss_d = torch.nn.functional.binary_cross_entropy_with_logits(
		input=d_real, target=ones
	)
	loss_d += torch.nn.functional.binary_cross_entropy_with_logits(
		input=d_fake, target=zeros
	)

	loss_g = torch.nn.functional.binary_cross_entropy_with_logits(
		input=d_fake, target=ones
	)

	return loss_d, loss_g

