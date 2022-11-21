import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')

class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'z' : h}

class GCNConv(nn.Module):
    def __init__(self, in_feats, out_feats,residual, activation):
        super(GCNConv, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)
        self.residual = residual
        if self.residual:
            if in_feats != out_feats:
                self.down = nn.Sequential(
                    nn.Linear(in_feats, out_feats,bias=False),
                )
            else:
                self.down = lambda x: x

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        g.ndata.pop('h')
        z_f = g.ndata.pop('z')
        if self.residual:
            z_f = z_f + self.down(feature)
        return z_f