import torch
import torch.nn as nn


class DuelingNetwork(torch.nn.Module):
	def __init__(self, input_dim, output_dim):
		super().__init__()
		c, h, w = input_dim

		self.online = self._build_model(c, output_dim)
		self.target = self._build_model(c, output_dim)
		self.target.load_state_dict(self.online.state_dict())

		# Q_target parameters are frozen
		for p in self.target.parameters():
			p.requires_grad = False

	def forward(self, input, model):
		if model == "online":
			return self.online(input)
		elif model == "target":
			return self.target(input)

	def _build_model(self, c, output_dim):
		model = nn.Sequential()

		# Convolutional layers
		model.add_module('conv1', nn.Conv2d(c, 32, kernel_size=8, stride=4))
		model.add_module('relu1', nn.ReLU())
		model.add_module('conv2', nn.Conv2d(32, 64, kernel_size=4, stride=2))
		model.add_module('relu2', nn.ReLU())
		model.add_module('conv3', nn.Conv2d(64, 64, kernel_size=3, stride=1))
		model.add_module('relu3', nn.ReLU())
		model.add_module('flatten', nn.Flatten())

		# Value and Advantage streams
		class DuelingHead(nn.Module):
			def __init__(self, input_features, output_dim):
				super().__init__()
				self.fc_common = nn.Linear(input_features, 512)
				self.relu = nn.ReLU()

				# Value stream
				self.value = nn.Sequential(
					nn.Linear(512, 256),
					nn.ReLU(),
					nn.Linear(256, 1)
				)

				# Advantage stream
				self.advantage = nn.Sequential(
					nn.Linear(512, 256),
					nn.ReLU(),
					nn.Linear(256, output_dim)
				)

			def forward(self, x):
				x = self.relu(self.fc_common(x))
				value = self.value(x)
				advantage = self.advantage(x)
				# avg advantage aggregation: Q = V + (A - mean(A))
				q = value + advantage - advantage.mean(dim=1, keepdim=True)
				return q

		model.add_module('dueling_head', DuelingHead(7 * 7 * 64, output_dim))
		return model


class Network(torch.nn.Module):
	"""mini CNN structure
	input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
	"""
	def __init__(self, input_dim, output_dim):
		super().__init__()
		c, h, w = input_dim

		if h != 84:
			raise ValueError(f"Expecting input height: 84, got: {h}")
		if w != 84:
			raise ValueError(f"Expecting input width: 84, got: {w}")

		self.online = self.__build_cnn(c, output_dim)
		self.target = self.__build_cnn(c, output_dim)
		self.target.load_state_dict(self.online.state_dict())

		# Q_target parameters are frozen.
		for p in self.target.parameters():
			p.requires_grad = False

	def forward(self, input, model):
		if model == "online":
			return self.online(input)
		elif model == "target":
			return self.target(input)

	def __build_cnn(self, c, output_dim):
		return nn.Sequential(
			nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4), # 32x20x20
			nn.ReLU(),
			nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), # 64x9x9
			nn.ReLU(),
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), # 64x7x7
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(3136, 512),
			nn.ReLU(),
			nn.Linear(512, output_dim),
		)