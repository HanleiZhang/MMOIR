import torch
import torch.nn as nn
import torch.nn.functional as F
from ..SubNets import text_backbones_map

class BERT_TEXT(nn.Module):

    def __init__(self, args):
        
        super(BERT_TEXT, self).__init__()
        text_backbone = text_backbones_map[args.text_backbone]
        if args.text_backbone == 'distilbert-base-nli-stsb-mean-tokens':
            self.text_subnet = text_backbone(args.pretrain_bert_model)[0].auto_model
        else:
            self.text_subnet = text_backbone(args)

    def forward(self, text_feats, video_feats, audio_feats):
        
        last_hidden_states = self.text_subnet(text_feats)
        features = last_hidden_states.mean(dim = 1)
        
        return features

class CCModel(nn.Module):

    def __init__(self, args):

        super(CCModel, self).__init__()
        self.backbone = BERT_TEXT(args)

        self.instance_projector = nn.Sequential(
            nn.Linear(args.feat_dim, args.feat_dim),
            nn.ReLU(),
            nn.Linear(args.feat_dim, args.feat_dim),
        )

        self.cluster_projector = nn.Sequential(
            nn.Linear(args.feat_dim, args.feat_dim),
            nn.ReLU(),
            nn.Linear(args.feat_dim, args.num_labels),
            nn.Softmax(dim=1),
        )
        
    def forward(self, text, video, audio):
        
        features = self.backbone(text, video, audio)
        return features
    
    def get_features(self, h_i, h_j):
        
        z_i = F.normalize(self.instance_projector(h_i), dim=1)
        z_j = F.normalize(self.instance_projector(h_j), dim=1)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        return z_i, z_j, c_i, c_j

    def forward_cluster(self, x):
     
        c = self.cluster_projector(x)
        c = torch.argmax(c, dim=1)
        return c
