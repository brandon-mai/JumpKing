#!/usr/env/bin python
#   
# Game Screen
#
from datetime import datetime
from pathlib import Path

import pygame 
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from environment import Environment
from King import King
from Babe import Babe
from Level import Levels
from Menu import Menus
from Start import Start
from RL.metric_logger import MetricLogger
from RL.DDQN import DDQN

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


class JKGame:
	""" Overall class to manga game aspects """
        
	def __init__(self, max_step=float('inf'), fps=60, cmap='rgb', span='full', edge_detect=False):

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

		# Pre-define for better performance
		self.cmap = cmap
		self.span = span
		self.edge_detect = edge_detect
		self.crop_size = int(int(os.environ.get("screen_height")) * 0.5)
		self.transforms = T.Compose([
			T.Resize((84, 84), antialias=True),
			T.Normalize(0, 255)
		])
		self.horizontal_kernel = torch.tensor([[-1, -1, -1],
										  [2, 2, 2],
										  [-1, -1, -1]], dtype=torch.float).unsqueeze(0).unsqueeze(0)

		self.vertical_kernel = torch.tensor([[-1, 2, -1],
										[-1, 2, -1],
										[-1, 2, -1]], dtype=torch.float).unsqueeze(0).unsqueeze(0)

		self.diagonal1_kernel = torch.tensor([[-1, -1, 2],
										 [-1, 2, -1],
										 [2, -1, -1]], dtype=torch.float).unsqueeze(0).unsqueeze(0)

		self.diagonal2_kernel = torch.tensor([[2, -1, -1],
										 [-1, 2, -1],
										 [-1, -1, 2]], dtype=torch.float).unsqueeze(0).unsqueeze(0)

	def reset(self):
		self.king.reset()
		self.levels.reset()
		# self.levels.current_level = 2
		os.environ["start"] = "1"
		os.environ["gaming"] = "1"
		os.environ["pause"] = ""
		os.environ["active"] = "1"
		os.environ["attempt"] = str(int(os.environ.get("attempt")) + 1)
		os.environ["session"] = "0"

		self.step_counter = 0
		done = False
		state = self.get_screen_array(cmap=self.cmap, span=self.span)

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

	def get_screen_array(self, cmap=None, span=None, edge_detect=False):
		# Use instance defaults if not specified
		cmap = cmap or self.cmap
		span = span or self.span
		cmap = 'gray' if cmap and edge_detect else cmap

		# Get surface array directly in correct orientation
		if cmap == 'gray':
			screen_arr = np.transpose(pygame.surfarray.array2d(self.game_screen))
			screen_arr = np.expand_dims(screen_arr, axis=0)  # Add channel dimension
		elif cmap == 'rgb':
			screen_arr = pygame.surfarray.array3d(self.game_screen)
			screen_arr = np.ascontiguousarray(screen_arr.transpose(2, 1, 0))
		else:
			raise ValueError(f"Unsupported color map: {cmap}")

		# Convert to tensor
		screen_arr = torch.from_numpy(screen_arr).float()

		if span == 'crop':
			# Get king position
			king_x, king_y = int(self.king.x), int(self.king.y)

			# Use pre-calculated crop size
			crop_size = self.crop_size
			crop_size_double = crop_size * 2

			# Create output tensor with exact size needed
			channels = screen_arr.shape[0]
			cropped = torch.zeros((channels, crop_size_double, crop_size_double),
								  dtype=torch.float, device=screen_arr.device)

			# Calculate crop boundaries
			h, w = screen_arr.shape[1], screen_arr.shape[2]
			top = king_y - crop_size
			left = king_x - crop_size

			# Compute valid source and destination regions
			src_top = max(0, top)
			src_left = max(0, left)
			src_bottom = min(h, top + crop_size_double)
			src_right = min(w, left + crop_size_double)

			# Calculate destination offsets
			dst_top = src_top - top
			dst_left = src_left - left
			h_slice = src_bottom - src_top
			w_slice = src_right - src_left

			# Only copy if there's actually content to copy
			if h_slice > 0 and w_slice > 0:
				cropped[:, dst_top:dst_top + h_slice, dst_left:dst_left + w_slice] = \
					screen_arr[:, src_top:src_bottom, src_left:src_right]

			screen_arr = cropped

		# Apply edge detection if requested
		if edge_detect:
			horizontal_kernel = self.horizontal_kernel.to(screen_arr.device)
			vertical_kernel = self.vertical_kernel.to(screen_arr.device)
			diagonal1_kernel = self.diagonal1_kernel.to(screen_arr.device)
			diagonal2_kernel = self.diagonal2_kernel.to(screen_arr.device)

			# Pad input to maintain dimensions
			padded_input = F.pad(screen_arr, (1, 1, 1, 1), mode='replicate')

			# Apply convolutions
			h_edges = F.conv2d(padded_input, horizontal_kernel)
			v_edges = F.conv2d(padded_input, vertical_kernel)
			d1_edges = F.conv2d(padded_input, diagonal1_kernel)
			d2_edges = F.conv2d(padded_input, diagonal2_kernel)

			# Combine edge responses
			edges = torch.clamp(torch.abs(h_edges) + torch.abs(v_edges) +
								torch.abs(d1_edges) + torch.abs(d2_edges), 0, 255)

			# Normalize to [0, 1] range
			screen_arr = edges / edges.max() if edges.max() > 0 else edges

		return self.transforms(screen_arr)

	def step(self, action):
		# ================ Record values before taking actions ================
		old_level = self.king.levels.current_level
		old_y = self.king.y

		# ============= Convert action int to stream of keypress =============
		match action:
			case 0:
				key_stream = ['right'] * 5
			case 1:
				key_stream = ['left'] * 5
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
				self._update_gamestuff(action=next(key_stream), mode='speed')
				self.update_av()
				continue
			except StopIteration:
				self._update_gamestuff(action=None, mode='speed')
				self.update_av()

			if self.move_available():
				self.step_counter += 1
				# state = [self.king.levels.current_level, self.king.x, self.king.y, self.king.jumpCount]
				state = self.get_screen_array(cmap=self.cmap, span=self.span, edge_detect=self.edge_detect)

				# =========================== Define the reward from environment ===========================
				if self.king.levels.current_level > old_level or (
						self.king.levels.current_level == old_level and self.king.y < old_y): # goes up
					reward = 10
				else: # goes down
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

				done = True if self.step_counter >= self.max_step else False
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

				# screen_arr = self.get_screen_array(cmap='gray', span='crop')  # (1, H, W)
				# screen_arr = screen_arr.squeeze(0)  # (H, W)

				# screen_arr = self.get_screen_array(cmap='gray', span='crop') # (C, H, W)
				# screen_arr = screen_arr.transpose(0, 1).transpose(1, 2) # (H, W, C)
				# plt.imshow(screen_arr)

				regular = self.get_screen_array(cmap='gray', span='crop', edge_detect=False)
				edge_detected = self.get_screen_array(cmap='gray', span='crop', edge_detect=True)

				fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
				regular_img = regular.squeeze(0).cpu().numpy()
				ax1.imshow(regular_img, cmap='gray')
				ax1.set_title('Original Image')
				ax1.axis('off')

				# Display edge-detected image
				edge_img = edge_detected.squeeze(0).cpu().numpy()
				ax2.imshow(edge_img, cmap='gray')
				ax2.set_title('Edge Detection')
				ax2.axis('off')

				plt.tight_layout()
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

	def _update_gamestuff(self, action=None, mode=None):

		self.levels.update_levels(self.king, self.babe, agentCommand=action, mode=mode)

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

	env = JKGame(max_step=500, fps=720, cmap='gray', span='crop', edge_detect=False)
	save_dir = Path("checkpoints") / datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
	save_dir.mkdir(parents=True)
	agent = DDQN(state_dim=(1, 84, 84), action_dim=8, save_dir=save_dir)
	logger = MetricLogger(save_dir)

	episodes = 1000

	for e in range(episodes):

		state = env.reset()

		# Play the game!
		while True:

			# Run agent on the state
			action = agent.act(state)

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
	Game = JKGame()
	Game.running()
	# train()