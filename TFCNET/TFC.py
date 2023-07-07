# encoding: utf-8
"""
@version: 1.0
@author: zxd3099
@file: BLSTM
@time: 2023-07-02 16:19
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)  # [max_len, d_model/2]
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: [x_len, batch_size, emb_size]
        :return: [x_len, batch_size, emb_size]
        """
        x = x + self.pe[:x.size(0), :]  # [x_len, batch_size, d_model]
        return self.dropout(x)


class TFC_Block(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 embedding_dim: int,
                 heads: int = 4):
        super(TFC_Block, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_size, bidirectional=True)
        self.norm = nn.LayerNorm(normalized_shape=[hidden_size])
        self.relu = nn.PReLU()
        self.pos_embedding = PositionalEncoding(d_model=embedding_dim)
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=heads)
        self.q_proj_weight = Parameter(torch.tensor(embedding_dim, embedding_dim))
        self.k_proj_weight = Parameter(torch.tensor(embedding_dim, embedding_dim))
        self.v_proj_weight = Parameter(torch.tensor(embedding_dim, embedding_dim))

    def forward(self, input: Tensor) -> Tensor:
        # [B, F, T, C] -> [BT, F, C]
        x = input.view(-1, input.shape[1], input.shape[3])
        # [BT, F, C] -> [F, BT, C]
        x1 = x.permute(1, 0, 2)

        hidden_state = torch.zeros(1*2, len(x), self.hidden_size)
        cell_state = torch.zeros(1*2, len(x), self.hidden_size)

        # output: [F, BT, hidden]
        output, (final_hidden_state, final_cell_state) = self.lstm(x1, (hidden_state, cell_state))
        # output = self.norm(torch.cat([x1, output], 2))
        output = self.norm(x1 + output)

        # [F, BT, C] -> [T, BF, C]
        x2 = output.view(input.shape[2], -1, input.shape[3])
        x3 = x2 + self.pos_embedding(x2)
        query = F.linear(x3, self.q_proj_weight)
        key = F.linear(x3, self.k_proj_weight)
        value = F.linear(x3, self.v_proj_weight)
        # y: [T, BF, C]
        y, attn_output_weights = self.attention(query, key, value)

        # [T, BF, C] -> [B, F, T, C]
        return self.relu(self.norm(y + x2)).view(*input.shape)
