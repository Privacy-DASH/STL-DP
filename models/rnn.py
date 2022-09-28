import os, random, json, time, argparse
import numpy as np
import pandas as pd

from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn

import torch
import torch.nn as nn


# Sequence to Sequence Module
class RnnEncoder(nn.Module):
    def __init__(self, input_dim, hid_dim=100, n_layers=2, dropout=0.5):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.proj = nn.Linear(input_dim, hid_dim)
        self.relu = nn.ReLU()
        
        self.rnn = nn.LSTM(hid_dim, hid_dim, n_layers, dropout = dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
            
        feat = self.relu(self.proj(x))   # [time stamp, batch size, in_units] >>> [time stamp, batch size, hidden dimension]
        outputs, (hidden, cell) = self.rnn(feat)
        
        '''
        outputs = [time stamp, batch size, hid dim * n directions]
        hidden = [n layers * n directions, batch size, hid dim]
        cell = [n layers * n directions, batch size, hid dim]
        outputs are always from the top hidden layer
        '''

        return hidden, cell

class RnnDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim=100, n_layers=2, dropout=0.5):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.proj = nn.Linear(input_dim, hid_dim)
        self.relu = nn.ReLU()
        
        self.rnn = nn.LSTM(hid_dim, hid_dim, n_layers, dropout = dropout)    
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        
        feat = self.relu(self.proj(input))   # [1, batch size, in_units] >>> [1, batch size, hidden dimension]
        output, (hidden, cell) = self.rnn(feat, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        
        return prediction, hidden, cell


class RnnSeq2Seq(nn.Module):
    """
    This module is based on recurrent neural network architecture
    """
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.encoder = RnnEncoder(args.in_units)  # input_dim, (hid_dim, n_layers, dropout)-default
        self.decoder = RnnDecoder(args.out_units, args.out_units) # input_dim, output_dim, (hid_dim, n_layers, dropout)-default

        assert self.encoder.hid_dim == self.decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.n_layers == self.decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    # def forward(self, src, trg, teacher_forcing_ratio = 0.0):
    def forward(self, src):
        
        src = src.permute(1, 0, 2)      # src = [src len, batch size]
        
        batch_size = src.shape[1]
        trg_len    = self.args.future_window
        output_dim = self.decoder.output_dim
        
        outputs = torch.zeros(trg_len, batch_size, output_dim).cuda(non_blocking=True) # tensor to store decoder outputs
        hidden, cell = self.encoder(src) #last hidden state of the encoder is used as the initial hidden state of the decoder
        
        start_token = torch.zeros(1, batch_size, output_dim).cuda(non_blocking=True)   # [time = 1, batch, feat]
        input = start_token 
        for t in range(1, trg_len):
            if len(input.shape) == 2:
                input = input.unsqueeze(0)
            
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output #place predictions in a tensor holding predictions for each token
            input = output
        
        return outputs.permute(1, 0, 2)    # prediction = [batch size, src len]