# encoding: utf-8
"""
@version: 1.0
@author: zxd3099
@file: __init__.py
@time: 2023-06-30 21:05
"""
import torch
import torch.nn as nn


a = torch.rand([1, 3, 4, 5])
b = torch.fft.ifft(a, dim=1)
print(a.shape)
print(b.shape)
print(a)
print(b)