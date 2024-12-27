import torch
import torch.nn.functional as F
from ..SubNets.FeatureNets import BERTEncoder, SubNet, RoBERTaEncoder, AuViSubNet
from ..SubNets.transformers_encoder.transformer import TransformerEncoder
# from .sampler import ConvexSampler
from torch import nn
from ..SubNets.AlignNets import AlignSubNet

activation_map = {'relu': nn.ReLU(), 'tanh': nn.Tanh()}
__all__ = ['umc']

class UMC(nn.Module):
    
    def __init__(self, args):

        super(UMC, self).__init__()
        
        self.args = args
        base_dim = args.base_dim
        
        self.num_heads = args.nheads
        self.attn_dropout = args.attn_dropout

        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.attn_mask = args.attn_mask
        
        self.text_embedding = BERTEncoder(args)
        
        self.text_layer = nn.Linear(args.text_feat_dim, base_dim)
        self.video_layer = nn.Linear(args.video_feat_dim, base_dim)
        self.audio_layer = nn.Linear(args.audio_feat_dim, base_dim)
         
        self.v_encoder = self.get_transformer_encoder(base_dim, args.encoder_layers_1)
        self.a_encoder = self.get_transformer_encoder(base_dim, args.encoder_layers_1)
        
        self.shared_embedding_layer = nn.Sequential(
            nn.GELU(),
            nn.Dropout(args.hidden_dropout_prob),
            nn.Linear(base_dim, base_dim),
        )

        self.fusion_layer = nn.Sequential(
            nn.GELU(),
            nn.Dropout(args.hidden_dropout_prob),
            nn.Linear(3 * base_dim, base_dim),
        )

    def get_transformer_encoder(self, embed_dim, layers):
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=layers,
                                  attn_dropout=self.attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)  
           
    def forward(self, text_feats, video_feats, audio_feats, mode = None): 

        video = video_feats.float()
        audio = audio_feats.float()
        text = self.text_embedding(text_feats) 

        video = self.video_layer(video)
        video = video.permute(1, 0, 2)
        video = self.v_encoder(video)[-1]
        
        audio = self.audio_layer(audio)
        audio = audio.permute(1, 0, 2)
        audio = self.a_encoder(audio)[-1] 
        
        text = text[:, 0]
        text = self.text_layer(text)        
        
        if mode == 'features':

            features = self.fusion_layer(torch.cat((text, audio, video), dim=1))
            return features

class UMCModel(nn.Module):

    def __init__(self, args):

        super(UMCModel, self).__init__()
        self.backbone = UMC(args)
        args.feat_dim = args.base_dim

        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.mlp_head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(args.hidden_dropout_prob),
            nn.Linear(args.base_dim, args.base_dim), 
        )
        

    def forward(self, text, video, audio, mode='train'):
        
        if mode == 'pretrain-mm':
            features = self.backbone(text, video, audio, mode='features')
            mlp_output = self.mlp_head(features)
            return mlp_output

        elif mode == 'train-mm':
            features = self.backbone(text, video, audio, mode='features')
            mlp_output = self.mlp_head(features)

            return features, mlp_output
        