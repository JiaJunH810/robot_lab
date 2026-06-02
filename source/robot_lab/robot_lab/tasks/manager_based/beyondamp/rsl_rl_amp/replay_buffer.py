import torch
import numpy as np

class ReplayBuffer:

    def __init__(self, obs_dim, buffer_size, device):
        self.generate_states = torch.zeros(buffer_size, obs_dim).to(device)
        self.generate_next_states = torch.zeros(buffer_size, obs_dim).to(device)
        self.expert_states = torch.zeros(buffer_size, obs_dim).to(device)
        self.expert_next_states = torch.zeros(buffer_size, obs_dim).to(device)

        self.buffer_size = buffer_size
        self.device = device

        self.step = 0
        self.num_samples = 0

    def insert(self, generate_states, generate_next_states, expert_states, expert_next_states):
        num_states = generate_states.shape[0]
        start_idx = self.step
        end_idx = self.step + num_states
        if end_idx > self.buffer_size:
            self.generate_states[self.step:self.buffer_size] = generate_states[:self.buffer_size - self.step]
            self.generate_next_states[self.step:self.buffer_size] = generate_next_states[:self.buffer_size - self.step]
            self.expert_states[self.step:self.buffer_size] = expert_states[:self.buffer_size - self.step]
            self.expert_next_states[self.step:self.buffer_size] = expert_next_states[:self.buffer_size - self.step]

            self.generate_states[:end_idx - self.buffer_size] = generate_states[self.buffer_size - self.step:]
            self.generate_next_states[:end_idx - self.buffer_size] = generate_next_states[self.buffer_size - self.step:]
            self.expert_states[:end_idx - self.buffer_size] = expert_states[self.buffer_size - self.step:]
            self.expert_next_states[:end_idx - self.buffer_size] = expert_next_states[self.buffer_size - self.step:]
        else:
            self.generate_states[start_idx:end_idx] = generate_states
            self.generate_next_states[start_idx:end_idx] = generate_next_states
            self.expert_states[start_idx:end_idx] = expert_states
            self.expert_next_states[start_idx:end_idx] = expert_next_states
        
        self.num_samples = min(self.buffer_size, max(end_idx, self.num_samples))
        self.step = (self.step + num_states) % self.buffer_size
    
    def feed_forward(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            sample_idxs = np.random.choice(self.num_samples, size=mini_batch_size)
            yield (self.generate_states[sample_idxs].to(self.device),
                   self.generate_next_states[sample_idxs].to(self.device),
                   self.expert_states[sample_idxs].to(self.device),
                   self.expert_next_states[sample_idxs].to(self.device))