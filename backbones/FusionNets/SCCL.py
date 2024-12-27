import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from ..SubNets import text_backbones_map

activation_map = {'relu': nn.ReLU(), 'tanh': nn.Tanh()}

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

class SCCLModel(nn.Module):
    
    def __init__(self, args):
        super(SCCLModel, self).__init__()
        self.backbone = BERT_TEXT(args)
        self.contrast_head = None
        self.cluster_centers = None

    def init_model(self, cluster_centers=None, alpha=1.0):

        # self.emb_size = self.bert.config.hidden_size
        self.emb_size = 768
        self.alpha = alpha
        
        # Instance-CL head
        self.contrast_head = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, 128))
        
        # Clustering head
        initial_cluster_centers = torch.tensor(
            cluster_centers, dtype=torch.float, requires_grad=True)
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, text, video, audio):
        return self.backbone.forward(text, video, audio)

    def get_cluster_prob(self, embeddings):
        norm_squared = torch.sum((embeddings.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

    def contrast_logits(self, embd1, embd2=None):
        feat1 = F.normalize(self.contrast_head(embd1), dim=1)
        if embd2 != None:
            feat2 = F.normalize(self.contrast_head(embd2), dim=1)
            return feat1, feat2
        else: 
            return feat1

  