import torch
import torch.nn as nn
from tensordict import TensorDict
from rsl_rl.networks import MLP

class Elastic(nn.Module):
    def __init__(self, 
                 obs: TensorDict,
                 obs_groups: dict[str, list[str]],
                 hidden_dims=[128, 64, 32], 
                 activation='elu'):
        super().__init__()
        self.obs_groups = obs_groups
        
        num_elastic_obs = 0
        for obs_group in obs_groups["elastic"]:
            num_elastic_obs += obs[obs_group].shape[-1]
        self.net = MLP(num_elastic_obs, 5, hidden_dims, activation)

    def forward(self, elastic_obs):
        return self.net(elastic_obs)
    
    def get_elastic_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["elastic"]]
        return torch.cat(obs_list, dim=-1)