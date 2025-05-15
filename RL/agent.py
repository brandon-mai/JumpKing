import torch
import numpy as np
import torch.nn as nn
from RL.network import DuelingNetwork
from RL.replay_buffer import ReplayBuffer


class PDD_DQN(object):
	"""Prioritized Dueling Double DQN agent."""
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
		self.gamma = 0.99 # discount factor for future rewards
		self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
		self.loss_fn = torch.nn.HuberLoss()

		self.n_step = 10
		self.memory = ReplayBuffer(
			capacity=100000,
			state_dim=state_dim,
			alpha=0.6,
			n_step=self.n_step,
			gamma=self.gamma,
		)
		self.beta = 0.4  # Initial beta for importance sampling ()
		self.beta_increment = 0  # Increment per step

		self.lambda1 = 1.0  # Weight for n-step double Q loss
		self.lambda2 = 1.0  # Weight for supervised classification loss
		self.lambda3 = 10e-5  # Weight for L2 regularization loss
		self.exp_margin = 0.8 # Expert margin for supervised classification loss

		self.burnin = 0  # min. experiences (steps) before training
		self.learn_every = 1  # no. of experiences between updates to Q_online
		self.sync_every = 10000  # no. of experiences between Q_target & Q_online sync (default: 10000)
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


	def cache(self, state, next_state, action, reward, done, demo=False):
		"""Store the experience to self.memory (replay buffer)."""
		def first_if_tuple(x):
			return x[0] if isinstance(x, tuple) else x

		state = first_if_tuple(state).__array__()
		next_state = first_if_tuple(next_state).__array__()

		self.memory.add(state, action, reward, next_state, done, demo)


	def recall(self):
		"""Retrieve a batch of experiences from memory with PER"""
		# Update beta for importance sampling
		self.beta = min(1.0, self.beta + self.beta_increment)

		# Sample from PER buffer
		samples = self.memory.sample(self.batch_size, self.beta)

		# Process 1-step data
		state = torch.FloatTensor(samples['obs']).to(self.device)
		next_state = torch.FloatTensor(samples['next_obs']).to(self.device)
		action = torch.LongTensor(samples['action']).to(self.device)
		reward = torch.FloatTensor(samples['reward']).to(self.device)
		done = torch.BoolTensor(samples['done']).to(self.device)
		demo = torch.BoolTensor(samples['demo']).to(self.device)
		
		# Process n-step data
		n_reward = torch.FloatTensor(samples['n_reward']).to(self.device)
		n_next_state = torch.FloatTensor(samples['n_next_obs']).to(self.device)
		n_done = torch.BoolTensor(samples['n_done']).to(self.device)
		
		weights = torch.FloatTensor(samples['weights']).to(self.device)
		indices = samples['indexes']

		return state, next_state, action, reward, done, n_reward, n_next_state, n_done, demo, weights, indices


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


	@torch.no_grad()
	def n_step_td_target(self, n_reward, n_next_state, n_done):
		"""Target net's n-step TD estimation, following DDQN's formula
		"""
		# Get best action from online network
		n_next_state_Q = self.net(n_next_state, model="online")
		n_best_action = torch.argmax(n_next_state_Q, axis=1)
		
		# Get Q-values from target network for those actions
		n_next_Q = self.net(n_next_state, model="target")[
			np.arange(0, self.batch_size), n_best_action
		]
		
		# For n-step return, we've already accumulated rewards with discount factors
		# in the replay buffer, so we just need to add the final bootstrapped Q-value
		discount = self.gamma ** self.n_step
		return (n_reward + (1 - n_done.float()) * discount * n_next_Q).float()


	def update_Q_online(self, td_estimate, td_target):
		loss = self.loss_fn(td_estimate, td_target)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		return loss.item()


	def sync_Q_target(self):
		self.net.target.load_state_dict(self.net.online.state_dict())


	def save(self, additional_data=None):
		"""
		Save the agent state to a checkpoint file.
		
		Args:
			additional_data: Optional dictionary with additional data to save
		"""
		save_path = (
			self.save_dir / f"checkpoint.chkpt"
		)
		
		# Create checkpoint dictionary
		checkpoint = {
			"model": self.net.state_dict(),
			"exploration_rate": self.exploration_rate,
			"curr_step": self.curr_step,
			"optimizer": self.optimizer.state_dict(),
			"buffer": self.memory,
		}
		
		# Add additional data if provided
		if additional_data:
			checkpoint.update(additional_data)
		
		# Save to file
		torch.save(checkpoint, save_path)
		print(f"Agent saved to {save_path} at step {self.curr_step}")
		
		return save_path


	def load(self, checkpoint_dir):
		"""
		Load agent state from a checkpoint dir, and continue training from there.
		
		Args:
			checkpoint_path: Path to the checkpoint file
		
		Returns:
			bool: True if loading was successful, False otherwise
		"""
		try:
			self.save_dir = checkpoint_dir
			checkpoint_path = checkpoint_dir / "checkpoint.ckkpt"
			if not checkpoint_path.exists():
				print(f"Checkpoint file not found at {checkpoint_path}")
				return False
			
			checkpoint = torch.load(checkpoint_path, map_location=self.device)
			
			# Load model weights
			self.net.load_state_dict(checkpoint["model"])
			
			# Load exploration rate if available
			if "exploration_rate" in checkpoint:
				self.exploration_rate = checkpoint["exploration_rate"]
				print(f"Loaded exploration rate: {self.exploration_rate:.4f}")
			
			# Load step counter if available
			if "curr_step" in checkpoint:
				self.curr_step = checkpoint["curr_step"]
				print(f"Loaded step counter: {self.curr_step}")
			
			# Load replay buffer if included in checkpoint
			if "buffer" in checkpoint and checkpoint["buffer"] is not None:
				self.memory = checkpoint["buffer"]
				print(f"Loaded replay buffer with {self.memory.size} experiences")
			
			# Load optimizer state if available
			if "optimizer" in checkpoint:
				self.optimizer.load_state_dict(checkpoint["optimizer"])
				print("Loaded optimizer state")
			
			# Additional metadata
			if "episode" in checkpoint:
				episode = checkpoint["episode"]
				print(f"Checkpoint was saved after episode {episode}")
			
			print(f"Successfully loaded checkpoint from {checkpoint_path}")
			return True
		
		except (FileNotFoundError, RuntimeError) as e:
			print(f"Failed to load checkpoint: {e}")
			return False


	def learn(self):
		if self.curr_step % self.sync_every == 0:
			self.sync_Q_target()

		if self.curr_step % self.save_every == 0:
			self.save()

		if self.curr_step < self.burnin:
			return None, None

		if self.curr_step % self.learn_every != 0:
			return None, None

		# Sample from memory with importance weights - include n-step data
		state, next_state, action, reward, done, n_reward, n_next_state, n_done, demo, weights, indices = self.recall()

		# Get TD Estimate (same for both 1-step and n-step)
		td_est = self.td_estimate(state, action)

		# Get 1-step TD Target
		td_tgt_1step = self.td_target(reward, next_state, done)
		
		# Get n-step TD Target
		td_tgt_nstep = self.n_step_td_target(n_reward, n_next_state, n_done)
		
		# Combine targets using lambda weighting
		td_tgt = td_tgt_1step + self.lambda1 * td_tgt_nstep
		
		# Calculate TD errors for updating priorities
		td_errors = torch.abs(td_est - td_tgt).detach().cpu().numpy()
		
		# Update priorities in replay buffer
		new_priorities = td_errors + 1e-6
		self.memory.update_priorities(indices, new_priorities)
		
		# 1-step TD loss with importance sampling
		td_loss = self.loss_fn(td_est, td_tgt)
		td_loss = (td_loss * weights).mean()

		# n-step TD loss with importance sampling
		nstep_loss = self.loss_fn(td_est, td_tgt_nstep)
		nstep_loss = (nstep_loss * weights).mean()

		# Large margin classification loss for expert demonstration
		margin_loss = torch.tensor(0.0, device=self.device)
		
		# Only calculate margin loss if there are demonstration samples in the batch
		if torch.any(demo):
			# Get the full Q-values for all actions
			q_values = self.net(state, model="online")
			
			# Create a mask for demonstration transitions
			demo_mask = demo.float()
			
			# Get one-hot encoding of expert actions
			action_one_hot = torch.zeros((self.batch_size, self.action_dim), device=self.device)
			action_one_hot.scatter_(1, action.unsqueeze(1), 1)
			
			# Calculate max[Q(s,a) + l(a,a_E)] where l(a,a_E) = margin if a ≠ a_E else 0
			# We add margin to all actions except the expert action
			margin_matrix = self.exp_margin * (1.0 - action_one_hot)  # margin for non-expert actions
			
			# Add margin to Q-values except for the expert action
			q_values_margin = q_values + margin_matrix
			
			# Get maximum Q-value (with margin) for each state
			max_q_values, _ = q_values_margin.max(dim=1)
			
			# Get Q-value for expert action
			expert_q_values = q_values.gather(1, action.unsqueeze(1)).squeeze()
			
			# Calculate loss: max[Q(s,a) + l(a,a_E)] - Q(s,a_E)
			# This encourages Q(s,a_E) to be higher than any other Q(s,a) by at least margin
			element_margin_loss = torch.clamp(max_q_values - expert_q_values, min=0.0)
			
			# Apply the loss only to demonstration samples
			margin_loss = (element_margin_loss * demo_mask * weights).sum() / (demo_mask.sum() + 1e-8)

		# L2 regularization loss
		l2_loss = torch.tensor(0.0, device=self.device)
		for param in self.net.parameters():
			l2_loss += torch.norm(param) ** 2
		
		# Total loss
		loss = td_loss + self.lambda1 * nstep_loss + self.lambda2 * margin_loss + self.lambda3 * l2_loss
		
		# Backpropagate loss
		self.optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=10.0)
		self.optimizer.step()
		
		return (td_est.mean().item(), loss.item())