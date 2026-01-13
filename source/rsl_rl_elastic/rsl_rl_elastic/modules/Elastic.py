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
        self.log_vars = nn.Parameter(torch.zeros(5))

    def forward(self, elastic_obs):
        return self.net(elastic_obs)
    
    def get_elastic_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["elastic"]]
        return torch.cat(obs_list, dim=-1)

    def compute_loss(self, pred, target):
        """
        根据 Kendall et al. (2018) 计算多任务不确定性损失
        Loss = 0.5 * sum( exp(-s) * (y - y_pred)^2 + s )
        其中 s = log_vars
        """
        squared_err = (pred - target) ** 2
        precision = torch.exp(-self.log_vars)
        loss = 0.5 * (precision * squared_err + self.log_vars)
        return loss.mean()
    