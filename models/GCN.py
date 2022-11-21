import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from models.GCNConv import GCNConv

class unit_GCN(nn.Module):
    def __init__(self, in_channels, out_channels,feat_drop=0,attn_drop=0,activation=None):
        super(unit_GCN, self).__init__()
        self.gat = GCNConv(in_feats=in_channels, out_feats=out_channels,residual=True,activation=activation)
        
    def forward(self, graph, x):
        #batch_size,nodes,frame,feature_size = x.shape
        #y = x.reshape(batch_size*nodes*frame,feature_size)
        y = self.gat(graph,x)
        #y = y.reshape(batch_size,nodes,frame,-1)
        
        return y

class GCN(nn.Module):
    def __init__(self, in_channels, nhidden, out_channels,feat_drop=0,attn_drop=0,activation=None):
        super(GCN, self).__init__()

        #self.l1 = unit_GAT(in_channels, in_channels,feat_drop,attn_drop,activation)
        self.l2 = unit_GCN(in_channels, out_channels,feat_drop,attn_drop,activation)

    def forward(self, graph, x):
        #x = self.l1(graph, x)
        x = self.l2(graph, x)
        return x
