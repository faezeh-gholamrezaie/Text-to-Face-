
import torch
import torch.nn as nn
import torch.optim as optim
from torch import  autograd
import torch.nn.functional as F
from torch.autograd import Variable

from torchvision import models
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import transformers
from sentence_transformers import SentenceTransformer

import pdb
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import os
import json
import time
import pprint
import importlib
import textwrap

import PIL
import io
import os, sys
import requests

import argparse
import easydict

from IPython.display import display, display_markdown
from Gan_1 import *

class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])
    
def conv1x1(in_planes, out_planes, bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=bias)


def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)

class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super().__init__()
        self.size = size
        self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, self.size, self.scale_factor, self.mode, self.align_corners)

    def extra_repr(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return info
class CNN_ENCODER(nn.Module):
    def __init__(self, nef):
        super(CNN_ENCODER, self).__init__()
        if True:
            self.nef = nef
        else:
            self.nef = 256  # define a uniform ranker

        model = models.inception_v3()
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        model.load_state_dict(model_zoo.load_url(url))
        for param in model.parameters(): # freeze inception model
            param.requires_grad = False
        print('Load pretrained model from ', url)
        # print(model)

        self.define_module(model)
        self.init_trainable_weights()

    def define_module(self, model):
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c

        self.emb_features = conv1x1(768, self.nef)
        self.emb_cnn_code = nn.Linear(2048, self.nef)

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        features = None
        # --> fixed-size input: batch x 3 x 299 x 299
        x = Upsample(size=(299, 299), mode='bilinear')(x)
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192

        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288

        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768

        # image region features
        features = x
        # 17 x 17 x 768

        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        # x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048

        # global image features
        cnn_code = self.emb_cnn_code(x) # nef

        if features is not None:
            features = self.emb_features(features) # 17 x 17 x nef
        return features, cnn_code

class BERT_CNN_ENCODER_RNN_DECODER(CNN_ENCODER):
    def __init__(self, emb_size, hidden_size,  nlayers=2, bidirectional=True, rec_unit='LSTM', dropout=0.5):
        """
        Based on https://github.com/komiya-m/MirrorGAN/blob/master/model.py
        :param emb_size: size of word embeddings
        :param hidden_size: size of hidden state of the recurrent unit
        :param vocab_size: size of the vocabulary (output of the network)
        :param rec_unit: type of recurrent unit (default=gru)
        """
        self.dropout = dropout
        self.nlayers = nlayers
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        __rec_units = {
            'GRU': nn.GRU,
            'LSTM': nn.LSTM,
        }
        assert rec_unit in __rec_units, 'Specified recurrent unit is not available'

        super().__init__(emb_size)

        self.hidden_linear = nn.Linear(emb_size, hidden_size)

        self.rnn = __rec_units[rec_unit](emb_size, hidden_size, num_layers=self.nlayers,
                         dropout=self.dropout)

        #self.out = nn.Linear(self.num_directions * hidden_size, 128)
        self.out = nn.Linear( hidden_size, emb_size)

    def forward(self, x, captions):
        # (bs x 17 x 17 x nef), (bs x nef)
        features, cnn_code = super().forward(x)
        # (bs x nef)
        cnn_hidden = self.hidden_linear(cnn_code)
        # (bs x hidden_size)

        #  (num_layers * num_directions, batch, hidden_size)
        #h_0 = cnn_hidden.unsqueeze(0).repeat(self.nlayers * self.num_directions, 1, 1) 
        h_0 = cnn_hidden.unsqueeze(0).repeat(self.nlayers , 1, 1)   
        c_0 = torch.zeros(h_0.shape).to(h_0.device)

  

        # bs x T x vocab_size
        # get last layer of bert encoder
        #text_embeddings, _ = self.encoder(captions, output_all_encoded_layers=False)
        text_embeddings = captions
        #text_embeddings = self.bert_linear(text_embeddings)
        # bs x T x emb_size
        output, (hn, cn) = self.rnn(text_embeddings, (h_0, c_0))
        # bs, T, hidden_size
        logits = self.out(output)
        # bs, T, vocab_size

        return output, features, cnn_code, logits    
