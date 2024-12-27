import torch.nn as nn
from torch import nn
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
    
class USNIDModel(nn.Module):

    def __init__(self, args):

        super(USNIDModel, self).__init__()
        self.backbone = BERT_TEXT(args)
        activation_map = {'relu': nn.ReLU(), 'tanh': nn.Tanh()}
        self.dense = nn.Linear(args.feat_dim, args.feat_dim)
        self.activation = activation_map[args.activation]
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.classifier = nn.Linear(args.feat_dim, args.num_labels)
        self.mlp_head = nn.Linear(args.feat_dim, args.num_labels)

    def forward(self, text, video, audio, feature_ext=False):
        features = self.backbone(text, video, audio)
        features = self.dense(features)
        pooled_output = self.activation(features)  
        pooled_output = self.dropout(features)
        mlp_outputs = self.mlp_head(pooled_output)
        
        if feature_ext:
            return features, mlp_outputs
        else:
            logits = self.classifier(pooled_output)
            return mlp_outputs, logits
