import torch
import numpy as np

class DiffNormalizer:
    def __init__(self, obs_dim, max_samples, device, min_diff=1e-4, clip=np.inf):
        self.device = device
        self.min_diff = min_diff
        self.clip = clip
        self.max_samples = max_samples

        self.count = 0
        self.mean_abs = torch.ones(obs_dim, device=device)

        self.new_count = 0
        self.new_sum_abs = torch.zeros(obs_dim, device=device)
    
    def record(self, x):
        if self.count >= self.max_samples:
            return
        assert len(x.shape) > len(self.mean_abs.shape)

        x = x.flatten(start_dim=0, end_dim=len(x.shape) - len(self.mean_abs.shape) - 1)
        self.new_count += x.shape[0]
        self.new_sum_abs += torch.sum(torch.abs(x), axis=0)

    def update(self):
        if self.count >= self.max_samples:
            return
        if self.new_count == 0:
            return
        old_sum_abs = self.mean_abs * self.count
        total_count = self.count + self.new_count

        self.mean_abs = (old_sum_abs + self.new_sum_abs) / total_count
        self.count = total_count

        self.new_count = 0
        self.new_sum_abs.zero_()

    def normalize(self, x):
        denom = torch.clamp(self.mean_abs, min=self.min_diff)
        norm_x = x / denom
        return torch.clamp(norm_x, -self.clip, self.clip)