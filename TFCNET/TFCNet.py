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
from MSF import MultiScaleBlock


class TFCNet(nn.Module):
    def __init__(self,
                 n_fft: int = 512,
                 hop_length1: int = 16,
                 window_length1: int = 64,
                 hop_length2: int = 256,
                 window_length2: int = 1024):
        super(TFCNet, self).__init__()
        self.corrector = Corrector()
        self.msf = MultiScaleBlock()

        self.n_fft = n_fft
        self.hop_length1 = hop_length1
        self.window_length1 = window_length1
        self.hop_length2 = hop_length2
        self.window_length2 = window_length2

    def forward(self, x, x1, x2):
        # x : [B, L]；t : [B, F, T, 2]
        t = torch.stft(x, self.n_fft, hop_length=self.hop_length1, win_length=self.window_length1)
        t_large = torch.stft(x, self.n_fft, hop_length=self.hop_length2, win_length=self.window_length2)
        t1 = torch.stft(x1, self.n_fft, hop_length=self.hop_length1, win_length=self.window_length1)
        t2 = torch.stft(x2, self.n_fft, hop_length=self.hop_length1, win_length=self.window_length1)

        # MSF
        t = self.msf(t, t_large)

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
