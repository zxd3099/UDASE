# encoding: utf-8
"""
@version: 1.0
@author: zxd3099
@file: MSF
@time: 2023-07-03 7:53
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, channels=12):
        super(BasicBlock, self).__init__()
        self.convIn = nn.Conv1d(in_channels, channels, kernel_size=1)
        self.convOut = nn.Conv1d(channels, in_channels, kernel_size=1)
        self.act = nn.ReLU()
        hidden_channels = channels // 3
        self.conv1 = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, kernel_size=1),
            nn.BatchNorm1d(hidden_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, kernel_size=1),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, kernel_size=1),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.Conv1d(hidden_channels, channels - 2 * hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels - 2 * hidden_channels)
        )

    def forward(self, x):
        """
        :param x: (N, F, L)
        :return:
        """
        # (N, F, L) -> (N, 12, L)
        x = self.act(self.convIn(x))
        # (N, 12, L) -> (N, 4, L)
        x1 = self.conv1(x)
        # (N, 12, L) -> (N, 4, L)
        x2 = self.conv2(x)
        # (N, 12, L) -> (N, 4, L)
        x3 = self.conv3(x)
        # (N, 4, L) -> (N, 12, L)
        y = torch.concat([x1, x2, x3], dim=1)
        return self.act(self.convOut(x + y))


class MultiScaleBlock(nn.Module):
    """
    Multi scale block
    """
    def __init__(self,
                 kernels: list = [5, 3, 3],
                 paddings: list = [0, 0, 19],
                 strides: list = [3, 7, 23],
                 block_channels: list = [257, 1028, 4097],
                 in_channels: list = [257, 1028, 257],
                 out_channels: list = [1028, 4097, 4097],
                 branches: int = 3
                 ):
        super(MultiScaleBlock, self).__init__()

        assert branches == len(kernels) and branches == len(paddings) and \
               branches == len(strides) and branches == len(in_channels) and \
               branches == len(out_channels) and branches == len(block_channels), "Dimension mismatch"

        self.blocks = []
        self.downSample = []
        for i in range(branches):
            self.blocks.append(BasicBlock(block_channels[i]))
            self.downSample.append(nn.Sequential(
                nn.Conv1d(in_channels[i], out_channels[i], kernel_size=kernels[i],
                          padding=paddings[i], stride=strides[i]),
                nn.BatchNorm1d(out_channels[i])
            ))

    def forward(self, x1, x3):
        """
        :param x1: speech signal processed by STFT1 (N, F1, T1, 2)
        :param x3: speech signal processed by STFT2 (N, F3, T3, 2)
        :return: the same shape as x1
        """
        # x1: (N, F1, T1, 2) -> (N, F1, T1 * 2)
        x1 = x1.view(x1.shape[0], x1.shape[1], -1)
        # x3: (N, F3, T3, 2) -> (N, F3, T3 * 2)
        x3 = x3.view(x3.shape[0], x3.shape[1], -1)

        x1 = self.blocks[0](x1)
        x2 = self.blocks[1](self.blocks[1](self.downSample[0](x1)))
        for i in range(5):
            x1 = self.blocks[0](x1)
            if i == 1:
                x3 += self.downSample[2](x1)
        x3 += self.downSample[1](x2)

        for i in range(3):
            x2 = self.blocks[1](x2)
            x3 = self.blocks[2](x3)

        # t2: (N, 1, F2, T2 * 2), t3: (N, 1, F3, T3 * 2)
        t1, t2, t3 = x1.unsqueeze(1), x2.unsqueeze(1), x3.unsqueeze(1)
        # y1: (N, F1, T1 * 2), y2: (N, F2, T2 * 2)
        y1 = F.interpolate(t2, size=(t1.shape[2:]), mode='bilinear') + F.interpolate(t3, size=(t1.shape[2:]), mode='bilinear')
        y2 = F.interpolate(t3, size=(t2.shape[2:]), mode='bilinear')
        y1 = y1.squeeze(1)
        y2 = y2.squeeze(1)

        y1 += x1
        y2 += x2 + self.downSample[0](x1)
        y3 = self.downSample[1](x2) + self.downSample[2](x1) + x3

        for i in range(2):
            y1 = self.blocks[0](y1)
            y2 = self.blocks[1](y2)
            y3 = self.blocks[2](y3)

        t1, t2, t3 = y1.unsqueeze(1), y2.unsqueeze(1), y3.unsqueeze(1)
        f1 = F.interpolate(t2, size=(t1.shape[2:]), mode='bilinear') + F.interpolate(t3, size=(t1.shape[2:]), mode='bilinear')
        f1 = f1.squeeze(1)
        f1 += y1

        return self.blocks[0](f1).view(x1.shape[0], x1.shape[1], -1, 2)
