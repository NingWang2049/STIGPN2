import torch.nn as nn
from .token import TokenEmbedding
from .position import PositionalEmbedding
from .segment import SegmentEmbedding
import torch

class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """
    
    def __init__(self, max_len, embed_size, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        #self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        #self.position = PositionalEmbedding(max_len=max_len,embed_size=embed_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, embed_size))
        self.segment_embedding = SegmentEmbedding(max_len=max_len,embed_size=embed_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sequence, segment_label):
        #x = self.position(sequence) + self.segment_embedding(segment_label).unsqueeze(0)
        x = self.pos_embedding# + self.segment_embedding(segment_label).unsqueeze(0)
        return x#self.dropout(x)
