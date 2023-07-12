# encoding: utf-8
"""
@version: 1.0
@author: zxd3099
@file: __init__.py
@time: 2023-06-30 21:05
"""
import torch
from MSF import MultiScaleBlock


a = torch.randn(1, 257, 247, 2)
b = torch.randn(1, 4097, 12, 2)

model = MultiScaleBlock()
c = model(a, b)
print(c.shape)
