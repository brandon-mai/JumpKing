import torch
import numpy as np
import torch.nn as nn
from RL.network import DuelingNetwork
from RL.replay_buffer import ReplayBuffer


class DDQN(object):
	def __init__(self, state_dim, action_dim, save_dir):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.save_dir = save_dir

		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.net = DuelingNetwork(input_dim=state_dim, output_dim=action_dim).float()
		self.net = self.net.to(device=self.device)

		self.exploration_rate = 1
		self.exploration_rate_decay = 0.99999975
		self.exploration_rate_min = 0.1
		self.curr_step = 0 # accumulated across episodes
		self.batch_size = 32
		self.gamma = 0.9
		self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
		self.loss_fn = torch.nn.SmoothL1Loss()

		self.memory = ReplayBuffer(capacity=100000, state_dim=state_dim, alpha=0.7)
		self.beta = 0.5  # Initial beta for importance sampling
		self.beta_increment = 0  # Increment per step

		self.burnin = 0  # min. experiences (steps) before training
		self.learn_every = 1  # no. of experiences between updates to Q_online
		self.sync_every = 1000  # no. of experiences between Q_target & Q_online sync
		self.save_every = 50000  # no. of experiences between saving NETWORK


	def act(self, state):
		"""Given a state, choose an epsilon-greedy action and update value of step.
		Args:
			state (LazyFrame): A single observation of the current state, dimension is (state_dim)
		Returns:
			int: An integer representing which action Mario will perform
		"""
		# EXPLORE
		if np.random.rand() < self.exploration_rate:
			action_idx = np.random.randint(self.action_dim)

		# EXPLOIT
		else:
			state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
			state = torch.tensor(state, device=self.device).unsqueeze(0)
			action_values = self.net(state, model="online")
			action_idx = torch.argmax(action_values, axis=1).item()

		# decrease exploration_rate
		self.exploration_rate *= self.exploration_rate_decay
		self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

		# increment step
		self.curr_step += 1
		return action_idx


	def cache(self, state, next_state, action, reward, done):
		"""Store the experience to self.memory (replay buffer)."""
		def first_if_tuple(x):
			return x[0] if isinstance(x, tuple) else x

		state = first_if_tuple(state).__array__()
		next_state = first_if_tuple(next_state).__array__()

		self.memory.add(state, action, reward, next_state, done)


	def recall(self):
		"""Retrieve a batch of experiences from memory with PER"""
		# Update beta for importance sampling
		self.beta = min(1.0, self.beta + self.beta_increment)

		# Sample from PER buffer
		samples = self.memory.sample(self.batch_size, self.beta)

		state = torch.FloatTensor(samples['obs']).to(self.device)
		next_state = torch.FloatTensor(samples['next_obs']).to(self.device)
		action = torch.LongTensor(samples['action']).to(self.device)
		reward = torch.FloatTensor(samples['reward']).to(self.device)
		done = torch.BoolTensor(samples['done']).to(self.device)
		weights = torch.FloatTensor(samples['weights']).to(self.device)
		indices = samples['indexes']

		return state, next_state, action, reward, done, weights, indices


	def td_estimate(self, state, action):
		"""In-training net's TD estimation
		"""
		current_Q = self.net(state, model="online")[
			np.arange(0, self.batch_size), action
		]  # Q_online(s,a)
		return current_Q


	@torch.no_grad()
	def td_target(self, reward, next_state, done):
		"""Target net's TD estimation, following DDQN's formula
		"""
		next_state_Q = self.net(next_state, model="online")
		best_action = torch.argmax(next_state_Q, axis=1)
		next_Q = self.net(next_state, model="target")[
			np.arange(0, self.batch_size), best_action
		]
		return (reward + (1 - done.float()) * self.gamma * next_Q).float()


	def update_Q_online(self, td_estimate, td_target):
		loss = self.loss_fn(td_estimate, td_target)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		return loss.item()


	def sync_Q_target(self):
		self.net.target.load_state_dict(self.net.online.state_dict())


	def save(self):
		save_path = (
			self.save_dir / f"net_{int(self.curr_step // self.save_every)}.chkpt"
		)
		torch.save(
			dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
			save_path,
		)
		print(f"Dairy Queen Net saved to {save_path} at step {self.curr_step}")


	def learn(self):
		if self.curr_step % self.sync_every == 0:
			self.sync_Q_target()

		if self.curr_step % self.save_every == 0:
			self.save()

		if self.curr_step < self.burnin:
			return None, None

		if self.curr_step % self.learn_every != 0:
			return None, None

		# Sample from memory with importance weights
		state, next_state, action, reward, done, weights, indices = self.recall()

		# Get TD Estimate
		td_est = self.td_estimate(state, action)

		# Get TD Target
		td_tgt = self.td_target(reward, next_state, done)

		# Calculate TD errors for updating priorities
		td_errors = torch.abs(td_est - td_tgt).detach().cpu().numpy()

		# Update PROPORTIONAL-BASED priorities in replay buffer
		new_priorities = td_errors + 1e-6
		self.memory.update_priorities(indices, new_priorities)

		# Apply importance weights to the loss
		elementwise_loss = self.loss_fn(td_est, td_tgt)
		loss = (elementwise_loss * weights).mean()

		# Backpropagate loss through Q_online
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		return (td_est.mean().item(), loss.item())