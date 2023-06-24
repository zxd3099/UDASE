# encoding: utf-8
"""
@version: 1.0
@author: zxd3099
@file: Encoder
@time: 2023-06-24 10:51
"""
import torch.nn as nn
import torch
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Convolutional Encoder Layer.

    Args:
        kernel_size: Length of filters.
        in_channels: Number of  input channels.
        out_channels: Number of output channels.

    Examples:

    >>> x = torch.randn(2, 1000)
    >>> encoder = Encoder(kernel_size=4, out_channels=64)
    >>> h = encoder(x)
    >>> h.shape # torch.Size([2, 64, 499])
    """

    def __init__(self,
                 kernel_size: int = 2,
                 out_channels: int = 64,
                 in_channels: int = 1):
        super(Encoder, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            groups=1,
            bias=False,
        )
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor):
        """Return the encoded output.

        Args:
            x: Input tensor with dimensionality [B, L].

        Returns:
            Encoded tensor with dimensionality [B, N, T_out].
            where B = Batchsize
                  L = Number of timepoints
                  N = Number of filters
                  T_out = Number of timepoints at the output of the encoder
        """
        # B x L -> B x 1 x L
        if self.in_channels == 1:
            x = torch.unsqueeze(x, dim=1)
        # B x 1 x L -> B x N x T_out
        x = self.conv1d(x)
        x = F.relu(x)

        return x
