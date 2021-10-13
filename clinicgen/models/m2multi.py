#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import torch
from torch import sigmoid
from torch.nn import Dropout, LayerNorm, Linear, Module
from torch.nn.functional import dropout, linear, relu, softmax
from torch.nn.init import normal_
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.transformer import _get_activation_fn, TransformerDecoderLayer, TransformerDecoder, TransformerEncoder
from torch.nn.parameter import Parameter
from clinicgen.models.m2transformer import M2Transformer
from clinicgen.models.image import ImageClassification

class M2MultiTransformer(M2Transformer):
    def __init__(self, embeddings, feat_dim=512, max_word=32, multi_image=1, layer_norm=False, num_memory=40,
                 num_enc_layers=6, num_dec_layers=6, teacher_forcing=False, image_model=None, image_pretrained=None,
                 finetune_image=False, image_finetune_epoch=None, rl_opts=None, word_idxs=None, device='gpu',
                 verbose=False):
        super(M2MultiTransformer, self).__init__(embeddings, feat_dim, max_word, multi_image, layer_norm, 
                 num_memory, num_enc_layers, num_dec_layers, teacher_forcing, image_model, image_pretrained,
                 finetune_image, image_finetune_epoch, rl_opts, word_idxs, device, verbose)
        # Transformer Encoder
        
#         encoder_layer = TransformerEncoderLayerWithMem(feat_dim, nhead=8, nmem=num_memory)
#         self.encoder = MeshedTransformerEncoder(encoder_layer, num_layers=num_enc_layers)
        # Transformer Decoder
#         decoder_layer = MeshedTransformerMaxDecoderLayer(feat_dim, nhead=8, nlayer_enc=num_enc_layers)
#         self.decoder = TransformerDecoder(decoder_layer, num_layers=num_dec_layers)


#     def deflatten_image(self, x):
#         if self.multi_image > 1:
#             x = x.view(int(x.shape[0] / self.multi_image), self.multi_image, self.encoder.num_layers, x.shape[2],
#                        x.shape[3])
#         return x

        print("finetuning image? ", finetune_image)
        self.image_feats, image_dim = ImageClassification.image_features(image_model, not finetune_image, True,
                                                                         image_pretrained, device)
        self.image_feats_secondary, image_dim_secondary = ImageClassification.image_features('resnet50', not finetune_image, True,
                                                                         None, device)
    
        self.double_image_proj = Linear(image_dim+image_dim_secondary, feat_dim)
    
    
    def encode_image(self, x):
        # CNN+Transformer features
        x = self.flatten_image(x)
#         print("encoding with 2 feature models")
        v, _ = self.image_features_with_mask_multi(x, self.image_feats, self.image_feats_secondary)
        # Merge multiple images
        v = self.deflatten_image(v)
        return v

    def image_features_with_mask_multi(self, x, model, model_secondary):
        mask = (x.detach().sum(dim=(1, 2, 3)) != 0).type(torch.float).unsqueeze(dim=-1).unsqueeze(dim=-1)
        if self.multi_image > 1:
            nz = mask.squeeze().nonzero().squeeze()
            x_nz = x[nz]
            if len(nz.shape) == 0:
                x_nz = x_nz.unsqueeze(dim=0)
        else:
            x_nz, nz = x, None
        # Image features
#         print("x_nz shape: ", x_nz.shape)
        
        x_nz1 = model(x_nz)
#         print("model output shape: ", x_nz1.shape)
        
        x_nz2 = model_secondary(x_nz)
#         print("model 2 output shape: ", x_nz2.shape)

        x_nz = torch.cat([x_nz1, x_nz2], dim=1)
#         print("combined shape: ", x_nz.shape)
        
        x_nz = x_nz.flatten(start_dim=-2, end_dim=-1)
        x_nz = x_nz.permute(0, 2, 1)
#         x_nz = relu(self.image_proj_l(x_nz))
        x_nz = relu(self.double_image_proj(x_nz))
        x_nz = self.dropout(x_nz)
        if self.layer_norm is not None:
            x_nz = self.layer_norm(x_nz)
#         print("pre-encoder x_nz shape: ", x_nz.shape)
        # Transformer encoder
        x_nz = x_nz.permute(1, 0, 2)
        x_nz = self.encoder(x_nz)
        x_nz = x_nz.permute(1, 2, 0, 3)
        xms = []
        if self.multi_image > 1:
            for i in range(self.encoder.num_layers):
                xm = x.new_zeros(x.shape[0], x_nz.shape[2], x_nz.shape[3])
                xm[nz] += x_nz[i]
                xms.append(xm)
        else:
            for i in range(self.encoder.num_layers):
                xms.append(x_nz[i])
        x = torch.stack(xms, dim=1)
        return x, mask
