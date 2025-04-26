#!/usr/env/bin python
#   
# Game Screen
#
from datetime import datetime
from pathlib import Path

import pygame 
import sys
import os
import inspect
import pickle
import numpy as np
from pygments.lexer import default

from environment import Environment
from spritesheet import SpriteSheet
from Background import Backgrounds
from King import King
from Babe import Babe
from Level import Levels
from Menu import Menus
from Start import Start
from MetricLogger import MetricLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import time
import matplotlib.pyplot as plt
import torchvision.transforms as T
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage


class CNN(torch.nn.Module):
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
			nn.Flatten(0),
			nn.Linear(3136, 512),
			nn.ReLU(),
			nn.Linear(512, output_dim),
		)


class DDQN(object):
	def __init__(self, state_dim, action_dim, save_dir):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.save_dir = save_dir

		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.net = CNN(self.state_dim, self.action_dim).float()
		self.net = self.net.to(device=self.device)

		self.exploration_rate = 1
		self.exploration_rate_decay = 0.99999975
		self.exploration_rate_min = 0.1
		self.curr_step = 0
		self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=torch.device("cpu")))
		self.batch_size = 32
		self.gamma = 0.9
		self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
		self.loss_fn = torch.nn.SmoothL1Loss()

		self.burnin = 1e4  # min. experiences before training
		self.learn_every = 3  # no. of experiences between updates to Q_online
		self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync
		self.save_every = 5e5  # no. of experiences between saving NETWORK

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
			print("we are exploiting, baby")
			state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
			state = torch.tensor(state, device=self.device).unsqueeze(0)
			action_values = self.net(state, model="online")
			action_idx = torch.argmax(action_values).item()

		# decrease exploration_rate
		self.exploration_rate *= self.exploration_rate_decay
		self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

		# increment step
		self.curr_step += 1
		return action_idx

	def cache(self, state, next_state, action, reward, done):
		"""Store the experience to self.memory (replay buffer).
        """
		def first_if_tuple(x):
			return x[0] if isinstance(x, tuple) else x

		state = first_if_tuple(state).__array__()
		next_state = first_if_tuple(next_state).__array__()

		state = torch.tensor(state)
		next_state = torch.tensor(next_state)
		action = torch.tensor([action])
		reward = torch.tensor([reward])
		done = torch.tensor([done])

		# self.memory.append((state, next_state, action, reward, done,))
		self.memory.add(
			TensorDict({"state": state, "next_state": next_state, "action": action, "reward": reward, "done": done},
					   batch_size=[]))

	def recall(self):
		"""Retrieve a batch of experiences from memory
        """
		batch = self.memory.sample(self.batch_size).to(self.device)
		state, next_state, action, reward, done = (batch.get(key) for key in
												   ("state", "next_state", "action", "reward", "done"))
		return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

	def td_estimate(self, state, action):
		"""In-training net's TD estimation
		"""
		if state.shape[1] != 1:
			state = state.permute(0, 3, 1, 2)

		current_Q = self.net(state, model="online")[
			np.arange(0, self.batch_size), action
		]  # Q_online(s,a)
		return current_Q

	@torch.no_grad()
	def td_target(self, reward, next_state, done):
		"""Target net's TD estimation, following DDQN's formula
		"""
		if next_state.shape[1] != 1:
			next_state = next_state.permute(0, 3, 1, 2)

		next_state_Q = self.net(next_state, model="online")
		best_action = torch.argmax(next_state_Q)
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

		# Sample from memory
		state, next_state, action, reward, done = self.recall()

		# Get TD Estimate
		td_est = self.td_estimate(state, action)

		# Get TD Target
		td_tgt = self.td_target(reward, next_state, done)

		# Backpropagate loss through Q_online
		loss = self.update_Q_online(td_est, td_tgt)

		return (td_est.mean().item(), loss)


class JKGame:
	""" Overall class to manga game aspects """
        
	def __init__(self, max_step=float('inf'), fps=60):

		pygame.init()

		self.environment = Environment()

		self.clock = pygame.time.Clock()

		self.fps = fps if isinstance(fps, int) else int(os.environ.get("fps"))
 
		self.bg_color = (0, 0, 0)

		self.screen = pygame.display.set_mode((int(os.environ.get("screen_width")) * int(os.environ.get("window_scale")), int(os.environ.get("screen_height")) * int(os.environ.get("window_scale"))), pygame.HWSURFACE|pygame.DOUBLEBUF)#|pygame.SRCALPHA)

		self.game_screen = pygame.Surface((int(os.environ.get("screen_width")), int(os.environ.get("screen_height"))), pygame.HWSURFACE|pygame.DOUBLEBUF)#|pygame.SRCALPHA)

		self.game_screen_x = 0

		pygame.display.set_icon(pygame.image.load("images\\sheets\\JumpKingIcon.ico"))

		self.levels = Levels(self.game_screen)

		self.king = King(self.game_screen, self.levels)

		self.babe = Babe(self.game_screen, self.levels)

		self.menus = Menus(self.game_screen, self.levels, self.king)

		self.start = Start(self.game_screen, self.menus)

		self.step_counter = 0
		self.max_step = max_step

		self.visited = {}

		pygame.display.set_caption('Jump King At Home XD')

	def reset(self):
		self.king.reset()
		self.levels.reset()
		os.environ["start"] = "1"
		os.environ["gaming"] = "1"
		os.environ["pause"] = ""
		os.environ["active"] = "1"
		os.environ["attempt"] = str(int(os.environ.get("attempt")) + 1)
		os.environ["session"] = "0"

		self.step_counter = 0
		done = False
		# state = [self.king.levels.current_level, self.king.x, self.king.y, self.king.jumpCount]
		state = self.get_screen_array()

		self.visited = {}
		self.visited[(self.king.levels.current_level, self.king.y)] = 1

		return state

	def update_av(self):
		self._update_gamescreen()
		self._update_guistuff()
		self._update_audio()
		pygame.display.update()

	def move_available(self):
		available = not self.king.isFalling \
					and not self.king.levels.ending \
					and (not self.king.isSplat or self.king.splatCount > self.king.splatDuration)
		return available

	def get_screen_array(self):
		screen_arr = pygame.surfarray.array2d(self.game_screen)
		screen_arr = screen_arr.transpose()[np.newaxis, ...]
		screen_arr = torch.tensor(screen_arr.copy(), dtype=torch.float)

		transforms = T.Compose([
			T.Resize((84, 84), antialias=True),
			T.Normalize(0, 255)
		])
		screen_arr = transforms(screen_arr).squeeze(0)
		return screen_arr

	def step(self, action):
		# ================ Record values before taking actions ================
		old_level = self.king.levels.current_level
		old_y = self.king.y
		#old_y = (self.king.levels.max_level - self.king.levels.current_level) * 360 + self.king.y

		# ============= Convert action int to stream of keypress =============
		match action:
			case 0:
				key_stream = ['right']
			case 1:
				key_stream = ['left']
			case 2:
				key_stream = ['space'] * 5 + ['right']
			case 3:
				key_stream = ['space'] * 5 + ['left']
			case 4:
				key_stream = ['space'] * 15 + ['right']
			case 5:
				key_stream = ['space'] * 15 + ['left']
			case 6:
				key_stream = ['space'] * 30 + ['right']
			case 7:
				key_stream = ['space'] * 30 + ['left']
			case _:
				key_stream = [action]
		key_stream = iter(key_stream)

		while True:
			self.clock.tick(self.fps)
			self._check_events()

			if os.environ["pause"]:
				self.update_av()
				continue

			try:
				self._update_gamestuff(action=next(key_stream))
				self.update_av()
				continue
			except StopIteration:
				self._update_gamestuff(action=None)
				self.update_av()

			if self.move_available():
				self.step_counter += 1
				# state = [self.king.levels.current_level, self.king.x, self.king.y, self.king.jumpCount]
				state = self.get_screen_array()

				# =========================== Define the reward from environment ===========================
				if self.king.levels.current_level > old_level or (
						self.king.levels.current_level == old_level and self.king.y < old_y):
					reward = 0
				else:
					current_key = (self.king.levels.current_level, self.king.y)
					old_key = (old_level, old_y)

					# Update current position visit count
					self.visited[current_key] = self.visited.get(current_key, 0) + 1

					# If current position has been visited less than previous, add previous count as base count
					old_pos_count = self.visited.get(old_key, 0)
					if self.visited[current_key] < old_pos_count:
						self.visited[current_key] = old_pos_count + 1

					# Negative reward based on visit count
					reward = -self.visited[current_key]

				done = True if self.step_counter > self.max_step else False
				return state, reward, done

	def running(self):
		"""
		play game with keyboard
		:return:
		"""
		self.reset()
		while True:
			#state = [self.king.levels.current_level, self.king.x, self.king.y, self.king.jumpCount]
			#print(state)
			self.clock.tick(self.fps)
			self._check_events()

			if os.environ["pause"]:
				self.update_av()
				continue

			self._update_gamestuff()

			self._update_gamescreen()
			self._update_guistuff()
			self._update_audio()
			pygame.display.update()

	def _check_events(self):

		for event in pygame.event.get():

			if event.type == pygame.QUIT:

				self.environment.save()

				self.menus.save ()

				pygame.quit()
				# screen_arr = pygame.surfarray.array2d(self.game_screen)
				# screen_arr = screen_arr.transpose()[np.newaxis, ...]
				# screen_arr = torch.tensor(screen_arr.copy(), dtype=torch.float)
				#
				# transforms = T.Compose([
				# 	T.Resize((84, 84), antialias=True),
				# 	T.Normalize(0, 255)
				# ])
				# screen_arr = transforms(screen_arr).squeeze(0)
				#
				# print(screen_arr.shape)
				# plt.imshow(screen_arr, cmap='gray')
				# plt.show()
				sys.exit()

			if event.type == pygame.KEYDOWN:

				self.menus.check_events(event)

				if event.key == pygame.K_c:

					if os.environ["mode"] == "creative":

						os.environ["mode"] = "normal"

					else:

						os.environ["mode"] = "creative"
					
			if event.type == pygame.VIDEORESIZE:

				self._resize_screen(event.w, event.h)

	def _update_gamestuff(self, action=None):

		self.levels.update_levels(self.king, self.babe, agentCommand=action)

	def _update_guistuff(self):

		if self.menus.current_menu:

			self.menus.update()

		if not os.environ["gaming"]:

			self.start.update()

	def _update_gamescreen(self):

		pygame.display.set_caption(f"Jump King At Home XD - {self.clock.get_fps():.2f} FPS")

		self.game_screen.fill(self.bg_color)

		if os.environ["gaming"]:

			self.levels.blit1()

		if os.environ["active"]:

			self.king.blitme()

		if os.environ["gaming"]:

			self.babe.blitme()

		if os.environ["gaming"]:

			self.levels.blit2()

		if os.environ["gaming"]:

			self._shake_screen()

		if not os.environ["gaming"]:

			self.start.blitme()

		self.menus.blitme()

		self.screen.blit(pygame.transform.scale(self.game_screen, self.screen.get_size()), (self.game_screen_x, 0))

	def _resize_screen(self, w, h):

		self.screen = pygame.display.set_mode((w, h), pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.SRCALPHA)

	def _shake_screen(self):

		try:

			if self.levels.levels[self.levels.current_level].shake:

				if self.levels.shake_var <= 150:

					self.game_screen_x = 0

				elif self.levels.shake_var // 8 % 2 == 1:

					self.game_screen_x = -1

				elif self.levels.shake_var // 8 % 2 == 0:

					self.game_screen_x = 1

			if self.levels.shake_var > 260:

				self.levels.shake_var = 0

			self.levels.shake_var += 1

		except Exception as e:

			print("SHAKE ERROR: ", e)

	def _update_audio(self):

		for channel in range(pygame.mixer.get_num_channels()):

			if not os.environ["music"]:

				if channel in range(0, 2):

					pygame.mixer.Channel(channel).set_volume(0)

					continue

			if not os.environ["ambience"]:

				if channel in range(2, 7):

					pygame.mixer.Channel(channel).set_volume(0)

					continue

			if not os.environ["sfx"]:

				if channel in range(7, 16):

					pygame.mixer.Channel(channel).set_volume(0)

					continue

			pygame.mixer.Channel(channel).set_volume(float(os.environ.get("volume")))


def train():
	use_cuda = torch.cuda.is_available()
	print(f"Using CUDA: {use_cuda}")
	print()

	env = JKGame(max_step=1000, fps=360)
	save_dir = Path("checkpoints") / datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
	save_dir.mkdir(parents=True)
	agent = DDQN(state_dim=(1, 84, 84), action_dim=8, save_dir=save_dir)
	logger = MetricLogger(save_dir)

	episodes = 10
	test_action = [2] * 5 + [0] + [2] * 15 + [1] + [2] * 25 + [0] + [0] * 1000
	test_action = iter(test_action)

	for e in range(episodes):

		state = env.reset()

		# Play the game!
		while True:

			# Run agent on the state
			action = agent.act(state)
			# action = next(test_action)

			# Agent performs action
			next_state, reward, done = env.step(action)

			# Remember
			agent.cache(state, next_state, action, reward, done)

			# Learn
			q, loss = agent.learn()

			# Logging
			logger.log_step(reward, loss, q)

			# Update state
			state = next_state

			# Check if end of game
			if done:
				break

		logger.log_episode()

		if (e % 1 == 0) or (e == episodes - 1):
			logger.record(episode=e, epsilon=agent.exploration_rate, step=agent.curr_step)

			
if __name__ == "__main__":
	# Game = JKGame()
	# Game.running()
	train()