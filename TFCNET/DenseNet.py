# encoding: utf-8
"""
@version: 1.0
@author: zxd3099
@file: DenseNet
@time: 2023-07-02 15:30
"""
import torch
import torch.nn as nn
from torch import Tensor


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation_rate,
                 kernel_size: int = 3,
                 stride: int = 1,
                 concat=True):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=((kernel_size - 1) // 2) * dilation_rate,
                              dilation=dilation_rate,
                              bias=False)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.PReLU()
        self.concat = concat

    def forward(self, x):
        out = self.relu(self.norm(self.conv(x)))
        if self.concat:
            return torch.cat([x, out], 1)
        else:
            return x


class DenseBlock(nn.Module):
    """
    The dilated dense block consists of 4 layers of 2-D convolution with dense connection\
    """
    def __init__(self,
                 in_channels: int = 128,
                 out_channels: int = 128,
                 num_layers: int = 4,
                 dilation_rates=[1, 2, 4, 8]):
        super(DenseBlock, self).__init__()
        assert len(dilation_rates) == num_layers, "The num of dilataion rates must equal num_layers"
        self.layer = self.make_layer(in_channels, out_channels, num_layers, dilation_rates)

    def make_layer(self, in_channels, out_channels, nb_layers, dilation_rates):
        layers = []
        for i in range(nb_layers):
            if i == nb_layers - 1:
                layers.append(BasicBlock(in_channels + i * out_channels, out_channels, dilation_rate=dilation_rates[i], concat=False))
            else:
                layers.append(BasicBlock(in_channels + i * out_channels, out_channels, dilation_rate=dilation_rates[i]))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # [B,F,T,C] -> [B,C,F,T]
        x = x.permute(0, 3, 1, 2)
        return self.layer(x).permute(0, 2, 3, 1)
