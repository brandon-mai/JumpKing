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
from tqdm import tqdm

from environment import Environment
from King import King
from Babe import Babe
from Level import Levels
from Menu import Menus
from Start import Start
from RL.metric_logger import MetricLogger
from RL.agent import PDD_DQN
from RL.replay_buffer import ReplayBuffer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


class JKGame:
	""" Overall class to manga game aspects """
        
	def __init__(self, max_step=float('inf'), training_mode=False, fps=60, cmap='rgb', span='full', edge_detect=False):

		pygame.init()

		self.training_mode = training_mode

		self.environment = Environment()
		self.clock = pygame.time.Clock()
		self.fps = fps if isinstance(fps, int) else int(os.environ.get("fps"))
		self.bg_color = (0, 0, 0)

		self.screen = pygame.display.set_mode((int(os.environ.get("screen_width")) * int(os.environ.get("window_scale")), 
												int(os.environ.get("screen_height")) * int(os.environ.get("window_scale"))), 
											pygame.HWSURFACE|pygame.DOUBLEBUF)
		pygame.display.set_icon(pygame.image.load("images\\sheets\\JumpKingIcon.ico"))

		self.game_screen = pygame.Surface((int(os.environ.get("screen_width")), int(os.environ.get("screen_height"))), pygame.HWSURFACE|pygame.DOUBLEBUF)
		self.game_screen_x = 0

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
		os.environ["recording"] = "0"

		self.step_counter = 0

		self.update_av()
		state = self.get_screen_array(cmap=self.cmap, span=self.span, edge_detect=self.edge_detect)

		self.visited = {}
		self.visited[(self.king.levels.current_level, self.king.y)] = 1

		return state

	def update_av(self):
		"""Update game screen, GUI, audio, and display."""
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
		# Record values before taking actions
		old_level = self.king.levels.current_level
		old_y = self.king.y

		# Convert action int to stream of keypress
		match action:
			case 0:
				key_stream = ['left'] * 10
			case 1:
				key_stream = ['space'] * 10 + ['left']
			case 2:
				key_stream = ['space'] * 15 + ['left']
			case 3:
				key_stream = ['space'] * 20 + ['left']
			case 4:
				key_stream = ['space'] * 25 + ['left']
			case 5:
				key_stream = ['space'] * 30 + ['leftspace']
			case 6:
				key_stream = ['right'] * 10
			case 7:
				key_stream = ['space'] * 10 + ['right']
			case 8:
				key_stream = ['space'] * 15 + ['right']
			case 9:
				key_stream = ['space'] * 20 + ['right']
			case 10:
				key_stream = ['space'] * 25 + ['right']
			case 11:
				key_stream = ['space'] * 30 + ['rightspace']
			case _:
				key_stream = [action]
		key_stream = iter(key_stream)

		# Fast version for training mode
		if self.training_mode:
			while True:
				self.clock.tick(self.fps)
				self._check_events()

				try:
					self._update_gamestuff(action=next(key_stream), mode='speed')
					continue
				except StopIteration:
					self._update_gamestuff(action=None, mode='speed')

				if self.move_available():
					self.step_counter += 1
					self.update_av()
					state = self.get_screen_array(cmap=self.cmap, span=self.span, edge_detect=self.edge_detect)

					# Define reward from environment
					if self.king.levels.current_level > old_level or (
							self.king.levels.current_level == old_level and self.king.y < old_y):
						reward = 0
					else:
						current_key = (self.king.levels.current_level, self.king.y)
						old_key = (old_level, old_y)
						self.visited[current_key] = self.visited.get(current_key, 0) + 1
						old_pos_count = self.visited.get(old_key, 0)
						if self.visited[current_key] < old_pos_count:
							self.visited[current_key] = old_pos_count + 1
						reward = -self.visited[current_key]

					done = True if self.step_counter >= self.max_step else False
					return state, reward, done
		
		# Original slower version for full movement gameplay
		else:
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
					state = self.get_screen_array(cmap=self.cmap, span=self.span, edge_detect=self.edge_detect)

					# Define reward from environment
					if self.king.levels.current_level > old_level or (
							self.king.levels.current_level == old_level and self.king.y < old_y):
						reward = 0
					else:
						current_key = (self.king.levels.current_level, self.king.y)
						old_key = (old_level, old_y)
						self.visited[current_key] = self.visited.get(current_key, 0) + 1
						old_pos_count = self.visited.get(old_key, 0)
						if self.visited[current_key] < old_pos_count:
							self.visited[current_key] = old_pos_count + 1
						reward = -self.visited[current_key]

					done = True if self.step_counter >= self.max_step else False
					return state, reward, done

	def running(self):
		"""
		play game with keyboard
		:return:
		"""
		self.reset()

		# Buffer for recording mode
		state_sample = self.get_screen_array(cmap=self.cmap, span=self.span, edge_detect=self.edge_detect)
		demo_buffer = ReplayBuffer(
			capacity=1000, state_dim=state_sample.shape,
			alpha=0.6, n_step=10, gamma=0.99,
		)

		# Try to load existing demonstrations if available
		demo_file = "demonstrations.pkl"
		demo_buffer.load_from_file(demo_file)

		old_state = None
		old_level = None
		old_y = None
		key_stream = iter([])
		action = None
		recording = False  # whether the game is recording an experience

		while True:
			self.clock.tick(self.fps)
			self._check_events()

			if os.environ["pause"]:
				self.update_av()
				continue

			if os.environ["recording"] == "0":
				self._update_gamestuff(mode='speed')
				self.update_av()
			else: # recording mode
				try: # execute key stream whenever one is available
					self._update_gamestuff(action=next(key_stream), mode='speed')
					self.update_av()
					continue

				except StopIteration: # key stream depleted
					if self.move_available() and recording: # idle after executing key stream, record results
						if self.king.levels.current_level > old_level or (
								self.king.levels.current_level == old_level and self.king.y < old_y):
							new_state = self.get_screen_array(cmap=self.cmap, span=self.span, edge_detect=self.edge_detect)
							
							if demo_buffer.size < demo_buffer.capacity:
								demo_buffer.add(
									old_state.cpu().numpy(),
									action,
									0, # reward for higher ground/level
									new_state.cpu().numpy(),
									False, # Not done
									demo=True, # expert demonstration,
									verbose=True
								)
							else:
								print("Buffer full, not adding experience")

						recording = False

					if self.move_available(): # idle, waiting for key stream command
						keys = pygame.key.get_pressed()

						if keys[pygame.K_f] and keys[pygame.K_1]:
							key_stream = ['left'] * 10
							action = 0
						elif keys[pygame.K_f] and keys[pygame.K_2]:
							key_stream = ['space'] * 10 + ['left']
							action = 1
						elif keys[pygame.K_f] and keys[pygame.K_3]:
							key_stream = ['space'] * 15 + ['left']
							action = 2
						elif keys[pygame.K_f] and keys[pygame.K_4]:
							key_stream = ['space'] * 20 + ['left']
							action = 3
						elif keys[pygame.K_f] and keys[pygame.K_5]:
							key_stream = ['space'] * 25 + ['left']
							action = 4
						elif keys[pygame.K_f] and keys[pygame.K_6]:
							key_stream = ['space'] * 30 + ['leftspace']
							action = 5

						elif keys[pygame.K_j] and keys[pygame.K_1]:
							key_stream = ['right'] * 10
							action = 6
						elif keys[pygame.K_j] and keys[pygame.K_2]:
							key_stream = ['space'] * 10 + ['right']
							action = 7
						elif keys[pygame.K_j] and keys[pygame.K_3]:
							key_stream = ['space'] * 15 + ['right']
							action = 8
						elif keys[pygame.K_j] and keys[pygame.K_4]:
							key_stream = ['space'] * 20 + ['right']
							action = 9
						elif keys[pygame.K_j] and keys[pygame.K_5]:
							key_stream = ['space'] * 25 + ['right']
							action = 10
						elif keys[pygame.K_j] and keys[pygame.K_6]:
							key_stream = ['space'] * 30 + ['rightspace']
							action = 11
						else:
							key_stream = []

						if keys[pygame.K_s]:
							demo_buffer.save_to_file(demo_file)
							print(f"Manually saved {demo_buffer.size} demonstrations")

						if key_stream:
							recording = True
							old_state = self.get_screen_array(
								cmap=self.cmap, span=self.span, edge_detect=self.edge_detect)
							old_level = self.king.levels.current_level
							old_y = self.king.y

						key_stream = iter(key_stream)

				self._update_gamestuff(action=None, mode='speed')
				self.update_av()

	def _check_events(self):
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				self.environment.save()
				self.menus.save ()
				pygame.quit()

				# screen_arr = self.get_screen_array(cmap='gray', span='crop')  # (1, H, W)
				# screen_arr = screen_arr.squeeze(0)  # (H, W)

				# screen_arr = self.get_screen_array(cmap='rgb', span='crop') # (C, H, W)
				# screen_arr = screen_arr.transpose(0, 1).transpose(1, 2) # (H, W, C)
				# plt.imshow(screen_arr)

				regular = self.get_screen_array(cmap='rgb', span='crop') # (C, H, W)
				regular = regular.transpose(0, 1).transpose(1, 2) # (H, W, C)

				plt.figure(figsize=(10, 10))
				plt.subplot(2, 2, 1)
				regular_img = regular.cpu().numpy()
				plt.imshow(regular_img)

				plt.subplot(2, 2, 2)
				# plt.plot(np.arange(0, 84), regular_img[:, 42, 0], label='R Channel', color='red')
				# plt.plot(np.arange(0, 84), regular_img[:, 42, 1], label='G Channel', color='green')
				# plt.plot(np.arange(0, 84), regular_img[:, 42, 2], label='B Channel', color='blue')
				plt.plot(regular_img[:, 42, 0], -np.arange(0, 84), label='R Channel', color='red')
				plt.plot(regular_img[:, 42, 1], -np.arange(0, 84), label='G Channel', color='green')
				plt.plot(regular_img[:, 42, 2], -np.arange(0, 84), label='B Channel', color='blue')

				plt.subplot(2, 2, 3)
				plt.plot(np.arange(0, 84), regular_img[42, :, 0], label='R Channel', color='red')
				plt.plot(np.arange(0, 84), regular_img[42, :, 1], label='G Channel', color='green')
				plt.plot(np.arange(0, 84), regular_img[42, :, 2], label='B Channel', color='blue')

				plt.tight_layout()
				# plt.show()

				sys.exit()

			if event.type == pygame.KEYDOWN:

				self.menus.check_events(event)
				# Creative mode
				if event.key == pygame.K_c:
					if os.environ["mode"] == "creative":
						os.environ["mode"] = "normal"
					else:
						os.environ["mode"] = "creative"

				# Recording mode
				if event.key == pygame.K_r:
					if os.environ.get("recording", "0") == "1":
						os.environ["recording"] = "0"
					else:
						print('recording mode ON')
						os.environ["recording"] = "1"
					
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


def play():
	""" Play the game with the keyboard."""
	Game = JKGame(cmap='rgb', span='crop', edge_detect=False)
	Game.running()


def pretrain():
    """Pretrain the agent with demonstrations."""
    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    
    # Create save directory
    save_dir = Path("checkpoints") / datetime.now().strftime("pretrain_%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)
    
    # Initialize agent
    agent = PDD_DQN(state_dim=(3, 84, 84), action_dim=12, save_dir=save_dir)
    logger = MetricLogger(save_dir)
    
    # Load expert demonstrations
    demo_file = "demonstrations.pkl"
    if not os.path.exists(demo_file):
        print(f"Error: Demonstration file '{demo_file}' not found!")
        return
    
    # Create a replay buffer with the right dimensions
    state_sample_shape = (3, 84, 84)  # RGB image
    demo_buffer = ReplayBuffer(
        capacity=1000,
        state_dim=state_sample_shape,
        alpha=0.6,
        n_step=10,
        gamma=0.99
    )
    
    # Load demonstrations
    if demo_buffer.load_from_file(demo_file):
        print(f"Successfully loaded demonstrations. Buffer has {demo_buffer.size} experiences.")
        # Set the agent's replay buffer to the loaded buffer
        agent.memory = demo_buffer
    else:
        print("Failed to load demonstrations, aborting pretraining.")
        return
    
    # Check if we have enough demos
    if demo_buffer.size < agent.batch_size:
        print(f"Error: Not enough demonstrations ({demo_buffer.size}) for pretraining! Need at least {agent.batch_size}.")
        return
    
    # Pretraining settings
    pretrain_steps = 50000
    print_interval = 5000
    save_interval = pretrain_steps
    
    print(f"Starting pretraining for {pretrain_steps} steps...")
    
    # During pretraining we want no exploration
    agent.exploration_rate = 0.01
    agent.exploration_rate_min = 0.01
    
    # Store pretrain metrics
    pretrain_losses = []
    pretrain_q_values = []
    
    # Perform pretraining
    for step in tqdm(range(pretrain_steps), desc="Pretraining", unit="step"):
		# No action taken so we have to manually increment the step
        agent.curr_step += 1

        # Force sampling of demo data during pretraining
        q, loss = agent.learn()
        
        # Store metrics
        if q is not None and loss is not None:
            pretrain_losses.append(loss)
            pretrain_q_values.append(q)
            logger.log_step(0, loss, q)  # Reward is 0 since we're not interacting with environment
        
        # Print progress
        if (step + 1) % print_interval == 0:
            avg_loss = np.mean(pretrain_losses[-print_interval:]) if pretrain_losses else 0
            avg_q = np.mean(pretrain_q_values[-print_interval:]) if pretrain_q_values else 0
            print(f"Pretraining Step {step+1}/{pretrain_steps} | Avg Loss: {avg_loss:.5f} | Avg Q-Value: {avg_q:.5f}")
            
        # Save model periodically
        # if (step + 1) % save_interval == 0:
        #     save_path = save_dir / f"pretrain_step_{step+1}.chkpt"
        #     torch.save(
        #         dict(
        #             model=agent.net.state_dict(),
        #             exploration_rate=agent.exploration_rate,
        #             pretrain_step=step+1,
        #             pretrain_total_steps=pretrain_steps
        #         ),
        #         save_path
        #     )
        #     print(f"Saved pretraining checkpoint to {save_path}")
    
    # Save final pretrained model
    final_save_path = save_dir / "pretrain_final.chkpt"
    torch.save(
        dict(
            model=agent.net.state_dict(),
            pretrained=True,
            pretrain_steps=pretrain_steps
        ),
        final_save_path
    )
    print(f"Pretraining complete! Final model saved to {final_save_path}")
    
    # Generate pretraining metrics plot
    if pretrain_losses:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(pretrain_losses)
        plt.title('Pretraining Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(pretrain_q_values)
        plt.title('Pretraining Q-Values')
        plt.xlabel('Step')
        plt.ylabel('Q-Value')
        
        plt.tight_layout()
        plt.savefig(save_dir / "pretrain_metrics.png")
        print(f"Pretraining metrics saved to {save_dir / 'pretrain_metrics.png'}")


def train():
	""" Train the agent with environment."""
	use_cuda = torch.cuda.is_available()
	print(f"Using CUDA: {use_cuda}")
	print()

	env = JKGame(max_step=500, training_mode=True, fps=900, cmap='rgb', span='crop', edge_detect=False)
	save_dir = Path("checkpoints") / datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
	save_dir.mkdir(parents=True)
	agent = PDD_DQN(state_dim=(3, 84, 84), action_dim=12, save_dir=save_dir)
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
	# play()
	pretrain()
	# train()

