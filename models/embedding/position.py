import torch.nn as nn
import torch
import math

class PositionalEmbedding(nn.Module):

    def __init__(self, max_len, embed_size):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, embed_size).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_size, 2).float() * -(math.log(10000.0) / embed_size)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        #pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:x.size(1),:]
