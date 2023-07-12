# encoding: utf-8
"""
@version: 1.0
@author: zxd3099
@file: TFConversion
@time: 2023-07-02 21:27
"""
import torch
import torch.nn as nn
from torch import Tensor


class FT_conversion(nn.Module):
    """
    Frequency-Time feature conversion
    """
    def __init__(self):
        super(FT_conversion, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: [B, F, T, C]
        :return: [B, (2F-1), T, C/2]
        """
        y_i, y_r = torch.split(x, split_size_or_sections=2, dim=-1)
        y_c = torch.complex(y_r, y_i)
        y_conj = torch.conj(y_c)
        y = torch.cat([y_c, y_conj], dim=1)
        return torch.real(torch.fft.ifft(y, dim=1))


class TF_conversion(nn.Module):
    """
    Frequency-Time feature conversion
    """

    def __init__(self):
        super(TF_conversion, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: [B, (2F-1), T, C/2]
        :return: [B, F, T, C]
        """
        y = torch.fft.fft(x)
        return torch.cat([x.imag, x.real], dim=-1)
