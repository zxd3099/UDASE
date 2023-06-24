# encoding: utf-8
"""
@version: 1.0
@author: zxd3099
@file: Decoder
@time: 2023-06-24 10:53
"""
import torch
import torch.nn as nn


class Decoder(nn.ConvTranspose1d):
    """A decoder layer that consists of ConvTranspose1d.

    Args:
        kernel_size: Length of filters.
        in_channels: Number of  input channels.
        out_channels: Number of output channels.

    Example
    ---------
    >>> x = torch.randn(2, 100, 1000)
    >>> decoder = Decoder(kernel_size=4, in_channels=100, out_channels=1)
    >>> h = decoder(x)
    >>> h.shape
    torch.Size([2, 1003])
    """

    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

    def forward(self, x):
        """Return the decoded output.

        Args:
            x: Input tensor with dimensionality [B, N, L].
            where, B = Batchsize,
                   N = number of filters
                   L = time points
        """

        if x.dim() not in [2, 3]:
            raise RuntimeError('{} accept 3/4D tensor as input'.format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))

        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)
        return x
