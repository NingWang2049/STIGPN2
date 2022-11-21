import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from models.ATTConv import ATTConv

class unit_ATT(nn.Module):
    def __init__(self, in_channels, out_channels,feat_drop=0,attn_drop=0,activation=None):
        super(unit_ATT, self).__init__()
        self.att = ATTConv(in_feats=in_channels, out_feats=out_channels,residual=True,activation=activation)
        
    def forward(self, graph, x):
        y = self.att(graph,x)
        
        return y

class ATT(nn.Module):
    def __init__(self, in_channels, nhidden, out_channels,feat_drop=0,attn_drop=0,activation=None,depth = 1):
        super(ATT, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth-1):
            self.layers.append(
                unit_ATT(in_channels, in_channels,feat_drop,attn_drop,activation)
            )
        self.layers.append(
                unit_ATT(in_channels, out_channels,feat_drop,attn_drop,activation)
            )
        
    def forward(self, graph, x):
        for ff in self.layers:
            x = ff(graph, x)
        return x