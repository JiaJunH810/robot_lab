# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
import torch.nn as nn

from rsl_rl.utils import resolve_nn_activation


class CNN1D(nn.Sequential):
    """
    Simplified 1D Convolutional Neural Network.
    Designed for time-series data processing (e.g., robot future trajectory).
    Input shape should be: (Batch, Input_Channels, Sequence_Length)
    """

    def __init__(
        self,
        input_length: int,            # 序列长度 (Time Steps, e.g., 10)
        input_channels: int,          # 输入特征维度 (Input Features, e.g., 172)
        output_channels: list[int],   # 每一层的输出通道数 (Hidden Sizes)
        kernel_size: list[int],       # 每一层的卷积核大小
        stride: list[int] | int = [1, 2],            # 每一层的步长
        padding: list[int] | int = 0, # Padding (可以是列表对应每一层，也可以是单一数值)
        activation: str = "elu",
        norm: str = "none",           # 'none', 'batch', 'layer'
        flatten: bool = True,
    ) -> None:
        super().__init__()

        # 辅助函数：处理参数如果是单个值则广播成列表
        num_layers = len(output_channels)
        def _to_list(arg, length):
            return arg if isinstance(arg, (list, tuple)) else [arg] * length

        kernel_size = _to_list(kernel_size, num_layers)
        stride = _to_list(stride, num_layers)
        padding = _to_list(padding, num_layers)

        activation_fn = resolve_nn_activation(activation)

        layers = []
        current_channels = input_channels
        current_length = input_length

        for i in range(num_layers):
            # 1. Convolution
            layers.append(nn.Conv1d(
                in_channels=current_channels,
                out_channels=output_channels[i],
                kernel_size=kernel_size[i],
                stride=stride[i],
                padding=padding[i]
            ))
            
            # 更新序列长度 L_out = floor((L_in + 2*padding - kernel) / stride + 1)
            current_length = math.floor(
                (current_length + 2 * padding[i] - kernel_size[i]) / stride[i] + 1
            )

            # 2. Normalization
            if norm == "batch":
                layers.append(nn.BatchNorm1d(output_channels[i]))
            elif norm == "layer":
                layers.append(nn.LayerNorm([output_channels[i], current_length]))

            # 3. Activation
            layers.append(activation_fn)

            current_channels = output_channels[i]

        # 4. Flatten
        if flatten:
            layers.append(nn.Flatten(start_dim=1))
            self._output_dim = current_channels * current_length
        else:
            self._output_dim = current_channels  # 如果不flatten，通常指最后的channel数，或者你需要自己处理(C, L)

        # Register layers
        for idx, layer in enumerate(layers):
            self.add_module(f"{idx}", layer)

        self.init_weights()

    @property
    def output_dim(self) -> int:
        """Returns the total flattened output dimension."""
        return self._output_dim

    def init_weights(self) -> None:
        """Xavier/Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)