from torch import nn
import torch
from tensordict import TensorDict

class Add(nn.Module):
    def __init__(self,
                 obs: TensorDict,
                 obs_groups: dict[str, list[str]],
                 hidden_dims: tuple[int] | list[int] = [1024, 512],
                 ):
        super().__init__()
        num_discriminator_obs = 0
        for obs_group in obs_groups["discriminator"]:
            num_discriminator_obs += obs[obs_group].shape[-1]
        self.in_feature = num_discriminator_obs

        layers = []
        in_size = num_discriminator_obs
        for units in hidden_dims:
            curr_layer = nn.Linear(in_size, units)
            torch.nn.init.zeros_(curr_layer.bias)
            layers.append(curr_layer)
            layers.append(nn.ReLU())
            in_size = units
        self.trunk = nn.Sequential(*layers)
        self.logits = nn.Linear(in_size, 1)
        self._init_output_weights()
    
    
    def _init_output_weights(self):
        init_output_scale = 1.0
        torch.nn.init.uniform_(self.logits.weight, -init_output_scale, init_output_scale)
        torch.nn.init.zeros_(self.logits.bias)
    
    def forward(self, x):
        h = self.trunk(x)
        return self.logits(h)
    
    def get_disc_logit_weights(self):
        return torch.flatten(self.logits.weight)

    def get_disc_weights(self):
        weights = []
        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                weights.append(torch.flatten(m.weight))
        weights.append(torch.flatten(self.logits.weight))
        return weights
    
    def combine_add_reward(self, obs: TensorDict, task_reward: torch.Tensor, normalizer):
        delta = obs["discriminator"]
        norm_delta = normalizer.normalize(delta)
        disc_logits = self.forward(norm_delta)
        disc_logits = disc_logits.squeeze(-1)

        prob = 1 / (1 + torch.exp(-disc_logits))
        disc_weight = 1
        disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=task_reward.device)))
        # disc_r = disc_weight * torch.clamp(1 - (1/4) * torch.square(disc_logits - 1), min=0)
        r = task_reward * 0 + disc_r * 1
        return r, disc_r