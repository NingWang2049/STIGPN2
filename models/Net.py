import os
import torch
import torch.nn as nn
import sys
import numpy as np
import dgl

from models.GAT import GAT
from models.GCN import GCN
from models.ATT import ATT
from models.embedding.BERTEmbedding import BERTEmbedding
from models.embedding.position import PositionalEmbedding

class V_SSTGCN(nn.Module):
    def __init__(self, args):
        super(V_SSTGCN, self).__init__()
        self.embedding_feature_dim = 256
        self.spatial_dim = 256
        self.nr_boxes = args.num_boxes
        self.nr_frames = args.num_frames//2
        self.k = args.k
        self.res_feat_dim = 2048
        self.preprocess_dim = 1024
        self.in_dim = (self.embedding_feature_dim + self.preprocess_dim)
        self.out_dim = 512
        self.act_classes = 174
        self.feat_drop = 0
        self.attn_drop = 0
        self.cls_dropout = 0.5

        self.SSTGC_params1 = self.SSTG(self.nr_boxes, self.nr_frames)
        self.SSTGC_params2 = self.SSTG(self.nr_boxes, self.nr_frames)
        self.SSTGC_params3 = self.SSTG(self.nr_boxes, self.nr_frames)

        ########################### Init GraphConv ###########################
        self.spatial_graph_evolution1 = ATT(self.in_dim,-1,self.out_dim,feat_drop=self.feat_drop,attn_drop=self.attn_drop,activation=nn.ReLU(),depth=1)
        self.inner_graph_evolution1 = ATT(self.in_dim,-1,self.out_dim,feat_drop=self.feat_drop,attn_drop=self.attn_drop,activation=nn.ReLU(),depth=1)
        
        self.spatial_graph_evolution2 = ATT(2*self.out_dim,-1,self.out_dim,feat_drop=self.feat_drop,attn_drop=self.attn_drop,activation=nn.ReLU(),depth=1)
        self.inner_graph_evolution2 = ATT(2*self.out_dim,-1,self.out_dim,feat_drop=self.feat_drop,attn_drop=self.attn_drop,activation=nn.ReLU(),depth=1)
        
        self.spatial_graph_evolution3 = ATT(2*self.out_dim,-1,self.out_dim,feat_drop=self.feat_drop,attn_drop=self.attn_drop,activation=nn.ReLU(),depth=1)
        self.inner_graph_evolution3 = ATT(2*self.out_dim,-1,self.out_dim,feat_drop=self.feat_drop,attn_drop=self.attn_drop,activation=nn.ReLU(),depth=1)

        #pre process
        self.appearence_preprocess = nn.Linear(self.res_feat_dim, self.preprocess_dim)
        self.coord_to_feature = nn.Sequential(
            nn.Linear(4, self.spatial_dim//2, bias=False),
            nn.BatchNorm1d(self.spatial_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(self.spatial_dim//2, self.spatial_dim, bias=False),
            nn.BatchNorm1d(self.spatial_dim),
            nn.ReLU()
        )

        self.occlusion_RNN = nn.RNN(input_size=self.in_dim, hidden_size=self.in_dim//2, num_layers=1, batch_first=True, bidirectional=True)
        self.occlusion_RNN.flatten_parameters()
        
        #fixed
        self.position = PositionalEmbedding(max_len=self.nr_frames*self.nr_boxes,embed_size=self.in_dim)
        #fixed T_S
        self.S_position = PositionalEmbedding(max_len=self.nr_boxes,embed_size=self.in_dim)
        self.T_position = PositionalEmbedding(max_len=self.nr_frames,embed_size=self.in_dim)
        #learningable T_S
        self.S_position = nn.Parameter(torch.randn(1, self.nr_boxes, self.in_dim))
        self.T_position = nn.Parameter(torch.randn(1, self.nr_frames, self.in_dim))

        self.classifier = nn.Sequential(
            nn.Linear(2*self.out_dim, 2*self.out_dim),
            # nn.BatchNorm1d(self.embedding_feature_dim),
            nn.Dropout(self.cls_dropout),
            nn.ReLU(inplace=True),
            nn.Linear(2*self.out_dim, 512), #self.embedding_feature_dim),
            nn.Dropout(self.cls_dropout),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.act_classes)
        )
    
    def SSTG(self, nr_boxes, nr_frames):
        k = self.k
        ######################## Build Graph ##########################
        #spatial_graph
        edge_list = [(0,n) for n in range(1,nr_boxes)]
        src, dst = tuple(zip(*edge_list))
        spatial_graph = dgl.graph((src, dst))
        spatial_graph = dgl.to_bidirected(spatial_graph)
        spatial_graph = spatial_graph.to('cuda')

        #inner_graph
        node_list = [x for x in range(nr_boxes)]
        node_frame_list = []
        for f_idx in range(nr_frames):
            temp = []
            for n_idx in node_list:
                temp.append(f_idx*nr_boxes+n_idx)
            node_frame_list.append(temp)
        edge_list = []
        for frame_idx in range(1, nr_frames-1):
            if frame_idx-k < 0:
                previous_idx = 0
            else:
                previous_idx = frame_idx-k
            previous_frame_list = node_frame_list[previous_idx:frame_idx]
            #remove human nodes
            previous_obj_nodes = []
            for item in previous_frame_list:
                previous_obj_nodes += [x for x in item[1:]]
            
            if frame_idx+k+1 > nr_frames:
                next_idx = nr_frames
            else:
                next_idx = frame_idx+k+1
            next_frame_list = node_frame_list[frame_idx+1:next_idx]
            #remove human nodes
            next_obj_nodes = []
            for item in next_frame_list:
                next_obj_nodes += [x for x in item[1:]]

            human_node_id = node_frame_list[frame_idx][0]

            edge_list += [(human_node_id,obj_node_id) for obj_node_id in previous_obj_nodes]
            edge_list += [(human_node_id,obj_node_id) for obj_node_id in next_obj_nodes]

        src, dst = tuple(zip(*edge_list))
        inner_graph = dgl.graph((src, dst))
        inner_graph = dgl.to_bidirected(inner_graph)
        inner_graph = inner_graph.to('cuda')

        return spatial_graph, inner_graph, nr_boxes, nr_frames

    def SSTGC(self, batch_size, SSTGC_params, spatial_graph_evolution, inner_graph_evolution, in_feats):
        spatial_graph, inner_graph, nr_boxes, nr_frames = SSTGC_params
        #batch spatial graph
        batch_spatial_graph = [spatial_graph for x in range(batch_size*nr_frames)]
        batch_spatial_graph = dgl.batch(batch_spatial_graph)
        #batch inner graph
        batch_inner_graph = [inner_graph for x in range(batch_size)]
        batch_inner_graph = dgl.batch(batch_inner_graph)
        
        spatial_feats = spatial_graph_evolution(batch_spatial_graph,in_feats)
        inner_feats = inner_graph_evolution(batch_inner_graph,in_feats)

        interaction_feats = torch.cat([spatial_feats, inner_feats],dim=1)

        return interaction_feats

    def forward(self, global_img_input, node_features, box_categories, box_input):
        batch_size = node_features.size(0)
        #spatial
        box_tensors_app = box_input.transpose(2, 1).contiguous()
        box_tensors_app = box_tensors_app.view(batch_size*self.nr_boxes*self.nr_frames, 4)
        spatial_feats = self.coord_to_feature(box_tensors_app)
        #appearence
        appearence_feats = self.appearence_preprocess(node_features.reshape(batch_size*self.nr_boxes*self.nr_frames,self.res_feat_dim))

        #appearence_spatial
        appearence_spatial_feats = torch.cat([spatial_feats, appearence_feats], dim=1)

        appearence_semantic_embedding = self.occlusion_RNN(appearence_spatial_feats.reshape(batch_size*self.nr_boxes,self.nr_frames,self.in_dim))[0]

        appearence_semantic_embedding = appearence_semantic_embedding.reshape(batch_size,self.nr_boxes, self.nr_frames, self.in_dim).transpose(2, 1)
        appearence_semantic_embedding = appearence_semantic_embedding.reshape(batch_size,self.nr_frames*self.nr_boxes, self.in_dim)
        appearence_semantic_embedding = appearence_semantic_embedding + self.position(appearence_semantic_embedding)
        appearence_semantic_embedding = appearence_semantic_embedding.reshape(batch_size*self.nr_frames*self.nr_boxes, self.in_dim)

        # spatial_semantic_embedding = spatial_semantic_embedding + self.T_position(spatial_semantic_embedding)
        # spatial_semantic_embedding = spatial_semantic_embedding.reshape(batch_size,self.nr_boxes, self.nr_frames, self.in_dim).transpose(2, 1)
        # spatial_semantic_embedding = spatial_semantic_embedding.reshape(batch_size*self.nr_frames,self.nr_boxes, self.in_dim)
        # spatial_semantic_embedding = spatial_semantic_embedding + self.S_position(spatial_semantic_embedding)
        # spatial_semantic_embedding = spatial_semantic_embedding.reshape(batch_size*self.nr_frames*self.nr_boxes, self.in_dim)
        
        # spatial_semantic_embedding = spatial_semantic_embedding + self.T_position
        # spatial_semantic_embedding = spatial_semantic_embedding.reshape(batch_size,self.nr_boxes, self.nr_frames, self.in_dim).transpose(2, 1)
        # spatial_semantic_embedding = spatial_semantic_embedding.reshape(batch_size*self.nr_frames,self.nr_boxes, self.in_dim)
        # spatial_semantic_embedding = spatial_semantic_embedding + self.S_position
        # spatial_semantic_embedding = spatial_semantic_embedding.reshape(batch_size*self.nr_frames*self.nr_boxes, self.in_dim)

        interaction_feats = self.SSTGC(batch_size, self.SSTGC_params1, self.spatial_graph_evolution1, self.inner_graph_evolution1, appearence_semantic_embedding)
        interaction_feats = self.SSTGC(batch_size, self.SSTGC_params2, self.spatial_graph_evolution2, self.inner_graph_evolution2, interaction_feats)
        #interaction_feats = self.SSTGC(batch_size, self.SSTGC_params3, self.spatial_graph_evolution3, self.inner_graph_evolution3, interaction_feats)

        interaction_feats = interaction_feats.reshape(batch_size,self.nr_frames,self.nr_boxes, 2*self.out_dim)
        
        human_node_feats = interaction_feats[:, :, 0, :]

        #h_cls_scores = torch.sum(torch.sum(self.classifier(interaction_feats), dim=1), dim=1)
        h_cls_scores = torch.sum(self.classifier(human_node_feats), dim=1)
        # obj_cls_scores = []
        # for b in range(batch_size):
        #     obj_feats = interaction_feats[b, :, 1: 1+num_objs[b], :]
        #     obj_scores = torch.mean(torch.sum(self.classifier(obj_feats), dim=0), dim=0)
        #     obj_cls_scores.append(obj_scores.unsqueeze(0))

        # o_cls_scores = torch.cat(obj_cls_scores, dim=0)
        
        return h_cls_scores# + o_cls_scores

class SSTGCN(nn.Module):
    def __init__(self, args):
        super(SSTGCN, self).__init__()
        self.embedding_feature_dim = 256
        self.spatial_dim = 256
        self.nr_boxes = args.num_boxes
        self.k = args.k
        self.nr_frames = args.num_frames//2
        self.in_dim = self.spatial_dim + self.embedding_feature_dim
        self.out_dim = 512
        self.act_classes = 174
        self.feat_drop = 0
        self.attn_drop = 0
        self.cls_dropout = 0.5

        self.SSTGC_params1 = self.SSTG(self.nr_boxes, self.nr_frames)
        self.SSTGC_params2 = self.SSTG(self.nr_boxes, self.nr_frames)
        self.SSTGC_params3 = self.SSTG(self.nr_boxes, self.nr_frames)

        ########################### Init GraphConv ###########################
        self.spatial_graph_evolution1 = ATT(self.in_dim,-1,self.out_dim,feat_drop=self.feat_drop,attn_drop=self.attn_drop,activation=nn.ReLU(),depth=1)
        self.inner_graph_evolution1 = ATT(self.in_dim,-1,self.out_dim,feat_drop=self.feat_drop,attn_drop=self.attn_drop,activation=nn.ReLU(),depth=1)
        
        self.spatial_graph_evolution2 = ATT(2*self.out_dim,-1,self.out_dim,feat_drop=self.feat_drop,attn_drop=self.attn_drop,activation=nn.ReLU(),depth=1)
        self.inner_graph_evolution2 = ATT(2*self.out_dim,-1,self.out_dim,feat_drop=self.feat_drop,attn_drop=self.attn_drop,activation=nn.ReLU(),depth=1)
        
        self.spatial_graph_evolution3 = ATT(2*self.out_dim,-1,self.out_dim,feat_drop=self.feat_drop,attn_drop=self.attn_drop,activation=nn.ReLU(),depth=1)
        self.inner_graph_evolution3 = ATT(2*self.out_dim,-1,self.out_dim,feat_drop=self.feat_drop,attn_drop=self.attn_drop,activation=nn.ReLU(),depth=1)

        #semantic_spatial_feature_extract
        self.category_embed_layer = nn.Embedding(3, self.embedding_feature_dim, padding_idx=0, scale_grad_by_freq=True)
        self.coord_to_feature = nn.Sequential(
            nn.Linear(4, self.spatial_dim//2, bias=False),
            nn.BatchNorm1d(self.spatial_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(self.spatial_dim//2, self.spatial_dim, bias=False),
            nn.BatchNorm1d(self.spatial_dim),
            nn.ReLU()
        )

        self.occlusion_RNN = nn.RNN(input_size=self.in_dim, hidden_size=self.in_dim//2, num_layers=1, batch_first=True, bidirectional=True)
        self.occlusion_RNN.flatten_parameters()
        
        #fixed
        self.position = PositionalEmbedding(max_len=self.nr_frames*self.nr_boxes,embed_size=self.in_dim)
        #fixed T_S
        self.S_position = PositionalEmbedding(max_len=self.nr_boxes,embed_size=self.in_dim)
        self.T_position = PositionalEmbedding(max_len=self.nr_frames,embed_size=self.in_dim)
        #learningable T_S
        self.S_position = nn.Parameter(torch.randn(1, self.nr_boxes, self.in_dim))
        self.T_position = nn.Parameter(torch.randn(1, self.nr_frames, self.in_dim))

        self.classifier = nn.Sequential(
            nn.Linear(2*self.out_dim, 2*self.out_dim),
            # nn.BatchNorm1d(self.embedding_feature_dim),
            nn.Dropout(self.cls_dropout),
            nn.ReLU(inplace=True),
            nn.Linear(2*self.out_dim, 512), #self.embedding_feature_dim),
            nn.Dropout(self.cls_dropout),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.act_classes)
        )
    
    def SSTG(self, nr_boxes, nr_frames):
        k = self.k
        ######################## Build Graph ##########################
        #spatial_graph
        edge_list = [(0,n) for n in range(1,nr_boxes)]
        src, dst = tuple(zip(*edge_list))
        spatial_graph = dgl.graph((src, dst))
        spatial_graph = dgl.to_bidirected(spatial_graph)
        spatial_graph = spatial_graph.to('cuda')

        #inner_graph
        node_list = [x for x in range(nr_boxes)]
        node_frame_list = []
        for f_idx in range(nr_frames):
            temp = []
            for n_idx in node_list:
                temp.append(f_idx*nr_boxes+n_idx)
            node_frame_list.append(temp)
        edge_list = []
        for frame_idx in range(1, nr_frames-1):
            if frame_idx-k < 0:
                previous_idx = 0
            else:
                previous_idx = frame_idx-k
            previous_frame_list = node_frame_list[previous_idx:frame_idx]
            #remove human nodes
            previous_obj_nodes = []
            for item in previous_frame_list:
                previous_obj_nodes += [x for x in item[1:]]
            
            if frame_idx+k+1 > nr_frames:
                next_idx = nr_frames
            else:
                next_idx = frame_idx+k+1
            next_frame_list = node_frame_list[frame_idx+1:next_idx]
            #remove human nodes
            next_obj_nodes = []
            for item in next_frame_list:
                next_obj_nodes += [x for x in item[1:]]

            human_node_id = node_frame_list[frame_idx][0]

            edge_list += [(human_node_id,obj_node_id) for obj_node_id in previous_obj_nodes]
            edge_list += [(human_node_id,obj_node_id) for obj_node_id in next_obj_nodes]

        src, dst = tuple(zip(*edge_list))
        inner_graph = dgl.graph((src, dst))
        inner_graph = dgl.to_bidirected(inner_graph)
        inner_graph = inner_graph.to('cuda')

        return spatial_graph, inner_graph, nr_boxes, nr_frames

    def SSTGC(self, batch_size, SSTGC_params, spatial_graph_evolution, inner_graph_evolution, in_feats):
        spatial_graph, inner_graph, nr_boxes, nr_frames = SSTGC_params
        #batch spatial graph
        batch_spatial_graph = [spatial_graph for x in range(batch_size*nr_frames)]
        batch_spatial_graph = dgl.batch(batch_spatial_graph)
        #batch inner graph
        batch_inner_graph = [inner_graph for x in range(batch_size)]
        batch_inner_graph = dgl.batch(batch_inner_graph)
        
        spatial_feats = spatial_graph_evolution(batch_spatial_graph,in_feats)
        inner_feats = inner_graph_evolution(batch_inner_graph,in_feats)

        interaction_feats = torch.cat([spatial_feats, inner_feats],dim=1)

        return interaction_feats

    def forward(self, global_img_input, node_features, box_categories, box_tensors):
        batch_size = box_tensors.size(0)
        
        #spatial features
        box_tensors = box_tensors.transpose(2, 1).contiguous()
        box_tensors = box_tensors.view(batch_size*self.nr_boxes*self.nr_frames, 4)
        spatial_feats = self.coord_to_feature(box_tensors).view(batch_size,self.nr_boxes,self.nr_frames, -1)
        #semantic features
        box_categories = box_categories.long()
        box_categories = box_categories.transpose(2, 1).contiguous()
        box_categories = box_categories.view(batch_size*self.nr_boxes*self.nr_frames)
        box_category_embeddings = self.category_embed_layer(box_categories).view(batch_size,self.nr_boxes,self.nr_frames, -1)
        #spatial semantic features
        spatial_semantic = torch.cat([spatial_feats,box_category_embeddings],dim=3)

        spatial_semantic_embedding = self.occlusion_RNN(spatial_semantic.reshape(batch_size*self.nr_boxes,self.nr_frames,self.in_dim))[0]

        spatial_semantic_embedding = spatial_semantic_embedding.reshape(batch_size,self.nr_boxes, self.nr_frames, self.in_dim).transpose(2, 1)
        spatial_semantic_embedding = spatial_semantic_embedding.reshape(batch_size,self.nr_frames*self.nr_boxes, self.in_dim)
        spatial_semantic_embedding = spatial_semantic_embedding + self.position(spatial_semantic_embedding)
        spatial_semantic_embedding = spatial_semantic_embedding.reshape(batch_size*self.nr_frames*self.nr_boxes, self.in_dim)

        # spatial_semantic_embedding = spatial_semantic_embedding + self.T_position(spatial_semantic_embedding)
        # spatial_semantic_embedding = spatial_semantic_embedding.reshape(batch_size,self.nr_boxes, self.nr_frames, self.in_dim).transpose(2, 1)
        # spatial_semantic_embedding = spatial_semantic_embedding.reshape(batch_size*self.nr_frames,self.nr_boxes, self.in_dim)
        # spatial_semantic_embedding = spatial_semantic_embedding + self.S_position(spatial_semantic_embedding)
        # spatial_semantic_embedding = spatial_semantic_embedding.reshape(batch_size*self.nr_frames*self.nr_boxes, self.in_dim)
        
        # spatial_semantic_embedding = spatial_semantic_embedding + self.T_position
        # spatial_semantic_embedding = spatial_semantic_embedding.reshape(batch_size,self.nr_boxes, self.nr_frames, self.in_dim).transpose(2, 1)
        # spatial_semantic_embedding = spatial_semantic_embedding.reshape(batch_size*self.nr_frames,self.nr_boxes, self.in_dim)
        # spatial_semantic_embedding = spatial_semantic_embedding + self.S_position
        # spatial_semantic_embedding = spatial_semantic_embedding.reshape(batch_size*self.nr_frames*self.nr_boxes, self.in_dim)

        interaction_feats = self.SSTGC(batch_size, self.SSTGC_params1, self.spatial_graph_evolution1, self.inner_graph_evolution1, spatial_semantic_embedding)
        interaction_feats = self.SSTGC(batch_size, self.SSTGC_params2, self.spatial_graph_evolution2, self.inner_graph_evolution2, interaction_feats)
        interaction_feats = self.SSTGC(batch_size, self.SSTGC_params3, self.spatial_graph_evolution3, self.inner_graph_evolution3, interaction_feats)

        interaction_feats = interaction_feats.reshape(batch_size,self.nr_frames,self.nr_boxes, 2*self.out_dim)
        
        human_node_feats = interaction_feats[:, :, 0, :]

        #h_cls_scores = torch.sum(torch.sum(self.classifier(interaction_feats), dim=1), dim=1)
        h_cls_scores = torch.sum(self.classifier(human_node_feats), dim=1)
        # obj_cls_scores = []
        # for b in range(batch_size):
        #     obj_feats = interaction_feats[b, :, 1: 1+num_objs[b], :]
        #     obj_scores = torch.mean(torch.sum(self.classifier(obj_feats), dim=0), dim=0)
        #     obj_cls_scores.append(obj_scores.unsqueeze(0))

        # o_cls_scores = torch.cat(obj_cls_scores, dim=0)
        
        return h_cls_scores# + o_cls_scores
