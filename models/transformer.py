# Reference Site: https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb

import os, random, json, time, argparse, math
import numpy as np
import pandas as pd

from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransSeq2Seq(nn.Module):
    
    def __init__(self, args, d_model: int = 512, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        """
        [src_len, batch_size, feature_size]
        """

        self.d_model = d_model
        self.args = args

        self.encoder_proj1 = nn.Linear(args.in_units, d_model)
        self.encoder_proj2 = nn.Conv2d(60, 20, 1)

        self.decoder_proj1 = nn.Linear(d_model, args.out_units)

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer, 6)
        self.decoder = nn.TransformerDecoder(decoder_layer, 6)

    def forward(self, src):
        batch_size = src.shape[0]

        projected = self.encoder_proj1(src)     # projecting in_units ---> d_model
        projected = projected.permute(1, 0, 2)  # [batch_size, seq_len, embedding_dim] >>> [seq_len, batch_size, embedding_dim]
        projected = self.pos_encoder(projected)

        feat = self.encoder(projected)
        feat = self.encoder_proj2(feat.unsqueeze(0)).squeeze(0)       
        start_token = torch.zeros(self.args.future_window, batch_size, self.d_model).cuda(non_blocking=True)
        
        prediction = self.decoder(start_token, feat)
        output = self.decoder_proj1(prediction).permute(1, 0, 2)
        return output