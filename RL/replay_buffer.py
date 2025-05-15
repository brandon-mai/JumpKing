import random
import numpy as np
import pickle


class ReplayBuffer:
    def __init__(self, capacity, state_dim, alpha=0.6, n_step=10, gamma=0.99):
        """Initialize replay buffer."""
        # Should use a power of $2$ for capacity because it simplifies the code and debugging
        self.capacity = capacity
        self.alpha = alpha
        self.n_step = n_step
        self.gamma = gamma

        # Demo data protection strategy
        self.writable_start = 0

        # PER data structures
        self.priority_sum = [0 for _ in range(2 * self.capacity)]
        self.priority_min = [float('inf') for _ in range(2 * self.capacity)]
        self.max_priority = 1.

        # n-step transition buffer
        self.n_step_buffer = []

        # Arrays for buffer
        self.data = {
            'obs': np.zeros(shape=(capacity, *state_dim), dtype=np.uint8),
            'action': np.zeros(shape=capacity, dtype=np.int32),
            'reward': np.zeros(shape=capacity, dtype=np.float32),
            'next_obs': np.zeros(shape=(capacity, *state_dim), dtype=np.uint8),
            'done': np.zeros(shape=capacity, dtype=np.bool),
            'n_reward': np.zeros(shape=capacity, dtype=np.float32),
            'n_next_obs': np.zeros(shape=(capacity, *state_dim), dtype=np.uint8),
            'n_done': np.zeros(shape=capacity, dtype=np.bool),
            'demo': np.zeros(shape=capacity, dtype=np.bool)
        }
        self.next_idx = self.writable_start
        self.size = 0


    def _get_n_step_info(self):
        """Calculate n-step return for the transition at the beginning of buffer"""
        # Get the first transition in the buffer
        reward, next_obs, done = self.n_step_buffer[0]['reward'], self.n_step_buffer[0]['next_obs'], self.n_step_buffer[0]['done']
        
        # Calculate n-step return by accumulating discounted rewards
        for idx in range(1, len(self.n_step_buffer)):
            if done:  # if the episode ends, no need to look further
                break
                
            curr_reward = self.n_step_buffer[idx]['reward']
            curr_done = self.n_step_buffer[idx]['done']
            curr_next_obs = self.n_step_buffer[idx]['next_obs']
            
            reward += (self.gamma ** idx) * curr_reward
            next_obs = curr_next_obs  # update next_observation to the furthest one
            done = curr_done  # update done flag
            
            if done:
                break
                
        return reward, next_obs, done


    def add(self, obs, action, reward, next_obs, done, demo=False, verbose=False):
        """Add experience to replay buffer with n-step return calculation"""
        # Add to n-step buffer
        self.n_step_buffer.append({
            'obs': obs,
            'action': action,
            'reward': reward,
            'next_obs': next_obs,
            'done': done,
            'demo': demo
        })
        
        # If we don't have enough transitions for n-step yet, just return
        if len(self.n_step_buffer) < self.n_step and not done:
            return
            
        # Get the earliest transition (for 1-step returns)
        obs_0 = self.n_step_buffer[0]['obs']
        action_0 = self.n_step_buffer[0]['action']
        reward_0 = self.n_step_buffer[0]['reward']
        next_obs_0 = self.n_step_buffer[0]['next_obs']
        done_0 = self.n_step_buffer[0]['done']
        demo_0 = self.n_step_buffer[0]['demo']
        
        # Calculate n-step returns
        n_reward, n_next_obs, n_done = self._get_n_step_info()
        
        # Get writable slot index in the buffer
        idx = self.next_idx
        
        # Store 1-step information
        self.data['obs'][idx] = obs_0
        self.data['action'][idx] = action_0
        self.data['reward'][idx] = reward_0
        self.data['next_obs'][idx] = next_obs_0
        self.data['done'][idx] = done_0
        self.data['demo'][idx] = demo_0
        
        # Store n-step information
        self.data['n_reward'][idx] = n_reward
        self.data['n_next_obs'][idx] = n_next_obs
        self.data['n_done'][idx] = n_done
        
        # Remove the earliest transition from n-step buffer
        self.n_step_buffer.pop(0)
        
        # Update buffer metadata
        self.next_idx = max((idx + 1) % self.capacity, self.writable_start) # must not be lower than writable_start
        self.size = min(self.capacity, self.size + 1)
        
        # Set priority (PER)
        priority_alpha = self.max_priority ** self.alpha
        self._set_priority_min(idx, priority_alpha)
        self._set_priority_sum(idx, priority_alpha)
        
        # If episode is done, clear the n-step buffer
        if done:
            self.n_step_buffer = []
        
        if verbose:
            print(f"Added experience #{self.size} to the buffer")


    def _set_priority_min(self, idx, priority_alpha):
        """
        #### Set priority in binary segment tree for minimum
        """

        # Leaf of the binary tree
        idx += self.capacity
        self.priority_min[idx] = priority_alpha

        # Update tree, by traversing along ancestors.
        # Continue until the root of the tree.
        while idx >= 2:
            # Get the index of the parent node
            idx //= 2
            # Value of the parent node is the minimum of it's two children
            self.priority_min[idx] = min(self.priority_min[2 * idx], self.priority_min[2 * idx + 1])


    def _set_priority_sum(self, idx, priority):
        """
        #### Set priority in binary segment tree for sum
        """

        # Leaf of the binary tree
        idx += self.capacity
        # Set the priority at the leaf
        self.priority_sum[idx] = priority

        # Update tree, by traversing along ancestors.
        # Continue until the root of the tree.
        while idx >= 2:
            # Get the index of the parent node
            idx //= 2
            # Value of the parent node is the sum of it's two children
            self.priority_sum[idx] = self.priority_sum[2 * idx] + self.priority_sum[2 * idx + 1]


    def _sum(self):
        """
        #### Get sum of all priorities
        """

        # The root node keeps the sum of all values
        return self.priority_sum[1]


    def _min(self):
        """
        #### Get minimum of all priorities
        """

        # The root node keeps the minimum of all values
        return self.priority_min[1]


    def find_prefix_sum_idx(self, prefix_sum):
        """
        #### Find largest idx such that Sum of priorities till idx is less than prefix_sum
        """

        # Start from the root
        idx = 1
        while idx < self.capacity:
            # If the sum of the left branch is higher than required sum
            if self.priority_sum[idx * 2] > prefix_sum:
                # Go to left branch of the tree
                idx = 2 * idx
            else:
                # Otherwise go to right branch and reduce the sum of left
                #  branch from required sum
                prefix_sum -= self.priority_sum[idx * 2]
                idx = 2 * idx + 1

        # We are at the leaf node. Subtract the capacity by the index in the tree
        # to get the index of actual value
        return idx - self.capacity


    def sample(self, batch_size, beta):
        """Sample from buffer with importance sampling"""
        # Your existing code to sample with importance sampling
        samples = {
            'weights': np.zeros(shape=batch_size, dtype=np.float32),
            'indexes': np.zeros(shape=batch_size, dtype=np.int32)
        }
        
        # Get sample indexes
        for i in range(batch_size):
            p = random.random() * self._sum()
            idx = self.find_prefix_sum_idx(p)
            samples['indexes'][i] = idx
        
        # Calculate probabilities and weights
        prob_min = self._min() / self._sum()
        max_weight = (prob_min * self.size) ** (-beta)
        
        for i in range(batch_size):
            idx = samples['indexes'][i]
            prob = self.priority_sum[idx + self.capacity] / self._sum()
            weight = (prob * self.size) ** (-beta)
            samples['weights'][i] = weight / max_weight
        
        # Get samples data - add n-step data
        for k, v in self.data.items():
            samples[k] = v[samples['indexes']]
        
        return samples


    def update_priorities(self, indexes, priorities):
        """
        ### Update priorities
        """

        for idx, priority in zip(indexes, priorities):
            # Set current max priority
            self.max_priority = max(self.max_priority, priority)

            # Calculate $p_i^\alpha$
            priority_alpha = priority ** self.alpha
            # Update the trees
            self._set_priority_min(idx, priority_alpha)
            self._set_priority_sum(idx, priority_alpha)


    def is_full(self):
        """
        ### Whether the buffer is full
        """
        return self.capacity == self.size
    

    def is_empty(self):
        """
        ### Whether the buffer is empty
        """
        return self.size == 0


    def save_to_file(self, filepath):
        """
        Save the replay buffer experiences to a file
        """
        # Create a dictionary with all the important buffer data
        buffer_data = {
            'data': self.data,
            'priority_sum': self.priority_sum,
            'priority_min': self.priority_min,
            'max_priority': self.max_priority,
            'next_idx': self.next_idx,
            'size': self.size
        }

        # Save to file using pickle
        with open(filepath, 'wb') as f:
            pickle.dump(buffer_data, f)

        print(f"Replay buffer saved to {filepath} with {self.size} experiences")


    def load_from_file(self, filepath):
        """
        Load replay buffer experiences from a file
        """
        if not self.is_empty():
            print("Expert data must be loaded when the buffer is empty")
            return False
        try:
            with open(filepath, 'rb') as f:
                buffer_data = pickle.load(f)

            # Restore the buffer state
            self.data = buffer_data['data']
            self.priority_sum = buffer_data['priority_sum']
            self.priority_min = buffer_data['priority_min']
            self.max_priority = buffer_data['max_priority']
            self.next_idx = buffer_data['next_idx']
            self.size = buffer_data['size']

            self.writable_start = self.size
            self.next_idx = self.writable_start

            print(f"Loaded {self.size} experiences into replay buffer from {filepath}")
            return True

        except (FileNotFoundError, IOError) as e:
            print(f"Failed to load replay buffer: {e}")
            return False


    def merge_buffer(self, other_buffer):
        """
        Merge another replay buffer into this one
        """
        # Add each experience from the other buffer
        for i in range(min(other_buffer.size, other_buffer.capacity)):
            idx = (other_buffer.next_idx - other_buffer.size + i) % other_buffer.capacity
            self.add(
                other_buffer.data['obs'][idx],
                other_buffer.data['action'][idx],
                other_buffer.data['reward'][idx],
                other_buffer.data['next_obs'][idx],
                other_buffer.data['done'][idx]
            )