# encoding: utf-8
"""
@version: 1.0
@author: zxd3099
@file: TFCNet
@time: 2023-07-02 23:10
"""
import torch
import torch.nn as nn
from corrector import Corrector


class TFCNet(nn.Module):
    def __init__(self,
                 n_fft: int = 512,
                 hop_length: int = 16,
                 window_length: int = 64):
        super(TFCNet, self).__init__()
        self.corrector = Corrector()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_length = window_length

    def forward(self, x, x1, x2):
        # x : [B, L]；t : [B, F, T, 2]
        t = torch.stft(x, self.n_fft, hop_length=self.hop_length, win_length=self.window_length)
        t1 = torch.stft(x1, self.n_fft, hop_length=self.hop_length, win_length=self.window_length)
        t2 = torch.stft(x2, self.n_fft, hop_length=self.hop_length, win_length=self.window_length)

        # input : [B, F, T, 6], output: [B, F, T, 4]
        input = torch.cat([t, t1, t2], -1)
        output = self.corrector(input)

        # split
        output1, output2 = torch.split(output, split_size_or_sections=2, dim=-1)
        output1 += t1
        output2 += t2

        y1 = torch.istft(output1, self.n_fft, hop_length=self.hop_length, win_length=self.window_length)
        y2 = torch.istft(output2, self.n_fft, hop_length=self.hop_length, win_length=self.window_length)
        return y1, y2
