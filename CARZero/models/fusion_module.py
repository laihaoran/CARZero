import torch
import torch.nn as nn
from .transformer import TransformerDecoderLayer, TransformerDecoder
import numpy as np
from sklearn import metrics

import ipdb


class Fusion(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        decoder_layer = TransformerDecoderLayer(cfg.model.fusion.d_model, cfg.model.fusion.H , 1024,
                                        0.1, 'relu',normalize_before=True)
        decoder_norm = nn.LayerNorm(cfg.model.fusion.d_model)
        self.decoder = TransformerDecoder(decoder_layer, cfg.model.fusion.N , decoder_norm,
                                  return_intermediate=False)

        # Learnable Queries
        #self.query_embed = nn.Embedding(config['num_queries'] ,self.d_model)
        self.dropout_feas = nn.Dropout(cfg.model.fusion.dropout)

        # Attribute classifier
        self.classifier = nn.Linear(cfg.model.fusion.d_model, cfg.model.fusion.state_prob)

    def forward(self, query_embed, features):
        features,ws = self.decoder(query_embed, features, 
            memory_key_padding_mask=None, pos=None, query_pos=None)
        out = self.dropout_feas(features)
        out = self.classifier(out).transpose(0,1) #B query Atributes
        return out
    
# query_embed = torch.ones([14, 768]) # bert output
# x = torch.ones([4, 16, 768]) # bs, p number, dim
# B= x.size(0)

# query_embed = query_embed.unsqueeze(1).repeat(1, B, 1)
# features = x.transpose(0,1) #patch_num b dim


# net = Fusion()
# net(query_embed, features)