from rsl_rl.networks import MLP
import torch
from torch import nn
from .cnn1d import CNN1D
from tensordict import TensorDict

class Actor(nn.Module):
    def __init__(self,
                 num_essence: int,
                 num_future_encoder: int,
                 num_action: int,
                 timesteps: int,
                 input_channels: int,
                 output_channels: list[int],
                 actor_hidden_dims: tuple[int] | list[int],
                 activation: str = "elu",
                 state_dependent_std: bool = False,
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

        if state_dependent_std:
            self.mlp = MLP(num_essence + num_future_encoder, [2, num_action], actor_hidden_dims, activation)
        else:
            self.mlp = MLP(num_essence + num_future_encoder, num_action, actor_hidden_dims, activation)

    def forward(self, obs: torch.Tensor):
        essence = obs[:, :self.num_essence]
        future = obs[:, self.num_essence:].view(obs.size(0), self.timesteps, -1)
        future_encoder = self.projection(self.cnn(future.permute(0, 2, 1)).contiguous())
        x = torch.cat([essence, future_encoder], dim=-1)
        return self.mlp(x)

class MoeActor(nn.Module):
    def __init__(self,
                 input_dim: int,
                 backbone_dim: int,
                 num_action: int,
                 actor_hidden_dims: tuple[int] | list[int],
                 activation: str = "elu",
                 num_experts: int = 3,
                 ):
        super().__init__()
        self.in_features = input_dim
        self.backbone = MLP(input_dim, backbone_dim * 2, actor_hidden_dims, activation)
        self.experts = nn.ModuleList([
            MLP(backbone_dim * 2, num_action, [backbone_dim * 2, backbone_dim], activation)
            for _ in range(num_experts)
        ])
        self.gate = nn.Sequential(
            MLP(backbone_dim * 2, num_experts, [backbone_dim * 2, backbone_dim], activation),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, obs):
        features = self.backbone(obs)
        weights = self.gate(features)
        print(weights)
        expert_actions = torch.stack([expert(features) for expert in self.experts], dim=1)
        weighted_action = torch.bmm(weights.unsqueeze(1), expert_actions).squeeze(1)
        return weighted_action