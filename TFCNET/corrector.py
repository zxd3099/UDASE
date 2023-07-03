# encoding: utf-8
"""
@version: 1.0
@author: zxd3099
@file: corrector
@time: 2023-06-30 21:05
"""
import torch.nn as nn
from torch import Tensor
from TFC import TFC_Block
from DenseNet import DenseBlock
from TFConversion import TF_conversion, FT_conversion


class Corrector(nn.Module):
    def __init__(self,
                 in_channels: int = 6,
                 emb_size: int = 64,
                 out_channels: int = 4,
                 repeat_time: int = 8):
        super(Corrector, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, emb_size, kernel_size=(1, 1), stride=1, bias=False)
        self.conv2 = nn.Conv2d(emb_size, out_channels, kernel_size=(1, 5), stride=1, padding=(0, 2), bias=False)
        self.dense_block = DenseBlock(emb_size, emb_size)
        self.layers = self.make_layers(repeat_time, emb_size)

    def make_layers(self, repeat_time, emb_size=64):
        layers = []
        for i in range(repeat_time):
            layers.append(TFC_Block())
            layers.append(FT_conversion())
            layers.append(TFC_Block(hidden_size=emb_size//2, embedding_dim=emb_size//2))
            layers.append(TF_conversion())
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.dense_block(self.conv1(x))
        return self.conv2(self.dense_block(self.layers(x1)))
