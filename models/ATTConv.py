import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_activation(name):
    if name=='relu':
        return nn.ReLU(inplace=True)
    elif name=='Tanh':
        return nn.Tanh()
    elif name=='Sigmoid':
        return nn.Sigmoid()
    elif name=='leaky_relu':
        return nn.LeakyReLU(0.2,inplace=True)
    else:
        assert(False), 'Not Implemented'

class attentionApplyModule(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0):
        super(attentionApplyModule, self).__init__()
        self.theta = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
        )
        if dropout:
            self.theta.add_module('dp',nn.Dropout(dropout))
        
        self.phi = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
        )
        if dropout:
            self.phi.add_module('dp',nn.Dropout(dropout))
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.theta[0].weight, gain=gain)
        nn.init.xavier_normal_(self.phi[0].weight, gain=gain)

    def forward(self, edge):
        x_i = edge.src['n_f']
        x_j = edge.dst['n_f']
        theta_x = self.theta(x_i).unsqueeze(1)
        phi_x = self.phi(x_j).unsqueeze(2)
        a_f = torch.matmul(theta_x,phi_x).squeeze(2)

        return {'a_f': a_f}

class nodeApplyModule(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0):
        super(nodeApplyModule, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
        )
        if dropout:
            self.fc.add_module('dp',nn.Dropout(dropout))
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc[0].weight, gain=gain)
    
    def forward(self, node):
        wh = self.fc(node.data['n_f'])

        return {'wh': wh}

class ATTConv(nn.Module):
    def __init__(self, in_feats,
                 out_feats,
                 feat_drop=0.,
                 attn_drop=0.,
                 residual=False,
                 activation=None):
        super(ATTConv, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.residual = residual
        self.activation = activation
        self.scale = out_feats ** -0.5

        self.nodeApplyfunc =  nodeApplyModule(self.in_feats,self.out_feats,self.feat_drop)
        self.attenApplyfunc =  attentionApplyModule(self.in_feats,self.out_feats,self.attn_drop)
        self.layernorm = nn.LayerNorm(self.out_feats)
        if self.residual:
            if in_feats != out_feats:
                self.down = nn.Sequential(
                    nn.Linear(in_feats, out_feats,bias=False),
                )
            else:
                self.down = lambda x: x

    def _message_func(self, edges):
        return {'wh_s': edges.src['wh'], 'wh_d': edges.dst['wh'], 'a_f': edges.data['a_f']}
    
    def _reduce_func(self, nodes):
        # calculate the features of virtual nodes
        alpha = F.softmax(nodes.mailbox['a_f'] * self.scale, dim=1)
        
        z_f = torch.sum(alpha * nodes.mailbox['wh_s'], dim=1)
        
        return {'z_f': z_f}
    
    def forward(self, graph,node_features):
        graph.ndata['n_f'] = node_features
        graph.apply_nodes(self.nodeApplyfunc,graph.nodes())
        graph.apply_edges(self.attenApplyfunc, graph.edges())
        graph.update_all(self._message_func, self._reduce_func)
        z_f = graph.ndata.pop('z_f')
        if self.residual:
            z_f = z_f + self.down(node_features)
        if self.activation:
            z_f = self.activation(z_f)
        z_f = self.layernorm(z_f)
        
        graph.ndata.pop('n_f')
        graph.ndata.pop('wh')
        graph.edata.pop('a_f')

        return z_f