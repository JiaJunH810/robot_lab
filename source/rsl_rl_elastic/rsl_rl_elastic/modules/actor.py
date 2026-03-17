from rsl_rl.networks import MLP, EmpiricalNormalization
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
                 num_action: int,
                 actor_hidden_dims: tuple[int] | list[int],
                 activation: str = "elu",
                 num_experts: int = 20,
                 ):
        super().__init__()
        self.obs_dim = input_dim
        self.experts = nn.ModuleList([
            MLP(input_dim, num_action, actor_hidden_dims, activation)
            for _ in range(num_experts)
        ])
        self.gate = nn.Sequential(
            MLP(input_dim, num_experts, actor_hidden_dims, activation),
            nn.Softmax(dim=-1)
        )
        # self.evaluate_counts = 0
        # self.all_entropy = 0.
    
    def forward(self, obs):
        weights = self.gate(obs)
        # self.calc_entropy(weights)
        expert_actions = torch.stack([expert(obs) for expert in self.experts], dim=1)
        weighted_action = torch.bmm(weights.unsqueeze(1), expert_actions).squeeze(1)
        return weighted_action

    def calc_entropy(self, weights):
        topk_weights, topk_indices = torch.topk(weights, k=8, dim=-1)

        topk_weights_norm = topk_weights / (torch.sum(topk_weights, dim=-1, keepdim=True) + 1e-12)
        entropy = -torch.sum(topk_weights_norm * torch.log(topk_weights_norm + 1e-12), dim=-1)
        self.evaluate_counts += 1
        self.all_entropy += entropy.mean()
        if self.evaluate_counts % 10 == 0:
            print(f"Top-8 Gating Entropy: {self.all_entropy / self.evaluate_counts:.4f}")

class VaeActor(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_vqvae: int,
                 num_action: int,
                 actor_hidden_dims: tuple[int] | list[int],
                 activation: str = "elu",
                 num_experts: int = 8,
                 ):
        super().__init__()
        self.obs_dim = input_dim
        self.code_dim = num_vqvae
        self.experts = nn.ModuleList([
            MLP(input_dim, num_action, actor_hidden_dims, activation)
            for _ in range(num_experts)
        ])
        self.gate = nn.Sequential(
            MLP(num_vqvae, num_experts, actor_hidden_dims, activation),
            nn.Softmax(dim=-1)
        )
        
        # self.evaluate_counts = 0
        # self.all_entropy = 0.
    
    def forward(self, obs, code):
        weights = self.gate(code)

        # self.calc_entropy(weights)
        expert_actions = torch.stack([expert(obs) for expert in self.experts], dim=1)
        weighted_action = torch.bmm(weights.unsqueeze(1), expert_actions).squeeze(1)
        return weighted_action
    
    # def calc_entropy(self, weights):
    #     topk_weights, topk_indices = torch.topk(weights, k=6, dim=-1)

    #     topk_weights_norm = topk_weights / (torch.sum(topk_weights, dim=-1, keepdim=True) + 1e-12)
    #     entropy = -torch.sum(topk_weights_norm * torch.log(topk_weights_norm + 1e-12), dim=-1)
    #     self.evaluate_counts += 1
    #     self.all_entropy += entropy.mean()
    #     if self.evaluate_counts % 10 == 0:
    #         print(f"Top-8 Gating Entropy: {self.all_entropy / self.evaluate_counts:.4f}")