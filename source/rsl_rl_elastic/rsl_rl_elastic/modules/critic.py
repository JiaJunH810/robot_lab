from rsl_rl.networks import MLP
import torch
from torch import nn
from .cnn1d import CNN1D
from tensordict import TensorDict
from rsl_rl.networks import MLP

class Critic(nn.Module):
    def __init__(self,
                 num_essence: int,
                 num_future_encoder: int,
                 timesteps: int,
                 input_channels: int,
                 output_channels: list[int],
                 critic_hidden_dims: tuple[int] | list[int],
                 activation: str = "elu",
                 ):
        super().__init__()
        self.num_essence = num_essence
        self.timesteps = timesteps
        self.in_features= num_essence + timesteps * input_channels

        self.cnn = CNN1D(
            input_length=timesteps,
            input_channels=input_channels,
            output_channels=output_channels,
            kernel_size=[3, 3],
            activation=activation
        )
        self.projection = nn.Linear(self.cnn.output_dim, num_future_encoder)

        self.mlp = MLP(num_essence + num_future_encoder, 1, critic_hidden_dims, activation)

    def forward(self, obs: torch.Tensor):
        essence = obs[:, :self.num_essence]
        future = obs[:, self.num_essence:].view(obs.size(0), self.timesteps, -1)
        future_encoder = self.projection(self.cnn(future.permute(0, 2, 1)).contiguous())
        x = torch.cat([essence, future_encoder], dim=-1)
        return self.mlp(x)

class MoeCritic(nn.Module):
    def __init__(self,
                 input_dim,
                 critic_hidden_dims: tuple[int] | list[int],
                 activation: str = "elu",
                 num_experts: int = 3,
                 ):
        super().__init__()

        self.experts = nn.ModuleList([
            MLP(input_dim, 1, critic_hidden_dims, activation)
            for _ in range(num_experts)
        ])

    def forward(self, obs):
        export_vals = torch.cat([export(obs) for export in self.experts], dim=-1)
        return export_vals