import torch
import numpy as np

class ReplayBuffer:

    def __init__(self, obs_dim, buffer_size, device):
        self.obs = torch.zeros(buffer_size, obs_dim).to(device)
        self.buffer_size = buffer_size
        self.device = device
        self.step = 0
        self.num_samples = 0
    
    def insert(self, obs):
        num_obs = obs.shape[0]
        start_idx = self.step
        end_idx = self.step + num_obs
        
        if end_idx > self.buffer_size:
            self.obs[self.step:self.buffer_size] = obs[:self.buffer_size - self.step]
            self.obs[:end_idx - self.buffer_size] = obs[self.buffer_size - self.step:]
        else:
            self.obs[start_idx:end_idx] = obs
        
        self.num_samples = min(self.buffer_size, max(end_idx, self.num_samples))
        self.step = (self.step + num_obs) % self.buffer_size
    
    def sample(self, batch_size):
        n = self.num_samples
        if n == 0:
            return None
        idx = torch.randint(0, n, (batch_size,), device=self.device)
        return self.obs[idx]