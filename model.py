import os, random, json, time, argparse
import numpy as np
import pandas as pd

from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn

import torch
import torch.nn as nn

from models import *


def call_by_name(*args):
    if args[0].model == 'Linear':
        model = Linear(args[0])
    elif args[0].model == 'SimpleDNN':
        model = SimpleDNN(args[0])
    elif args[0].model == 'RNNSeq2Seq':
        model = RnnSeq2Seq(args[0])
    elif args[0].model == 'TransSeq2Seq':
        model = TransSeq2Seq(args[0])
    else:
        print("Model is not implemented!")
        exit()
    return model

class Linear(torch.nn.Module):
    def __init__(self, args):
        super(Linear, self).__init__()

        self.args = args
        self.input_size  = int(args.past_window * args.in_units) 
        self.output_size = int(args.future_window * args.out_units)
        self.linear_regression = nn.Linear(self.input_size, self.output_size)

    def forward(self, x):
        """
        inputs: [Batch, seq_len, inputs]
        """
        batch_size = x.shape[0]
        seq_len = self.args.future_window

        x = x.flatten(1)
        out = self.linear_regression(x).reshape(batch_size, seq_len, -1)
        return out

class SimpleDNN(torch.nn.Module):
    def __init__(self, args):
        super(SimpleDNN, self).__init__()
        self.args = args
        input_size  = int(args.past_window * args.in_units)
        output_size = int(args.future_window * args.out_units)

        self.linear1 = nn.Linear(input_size, input_size//2)
        self.linear2 = nn.Linear(input_size//2, input_size//2)
        self.linear3 = nn.Linear(input_size//2, input_size//4)
        self.linear4 = nn.Linear(input_size//4, output_size)

    def forward(self, x):
        """
        inputs: [Batch, seq_len, inputs]
        """
        batch_size = x.shape[0]
        seq_len = self.args.future_window

        x = x.flatten(1)
        out1 = nn.ReLU()(self.linear1(x))
        out2 = nn.ReLU()(self.linear2(out1))
        out3 = nn.ReLU()(self.linear3(out2))
        out4 = nn.ReLU()(self.linear4(out3)).reshape(batch_size, seq_len, -1)

        return out4
