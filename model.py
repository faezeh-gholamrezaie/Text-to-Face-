
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
from transformer import *

batch_size=128
sample_data_loader = torch.utils.data.DataLoader(
    sample_dataset,
    batch_size=batch_size,
    shuffle=False
)
embeddings_by_handmade_model = []
right_image_batch = []
right_image_batch128 = []
right_image_batch256 = []
Gaussian_Noise = False

def features_to_device(features, device):
    for feature_name in features:
        if hasattr(features[feature_name], 'keys'):
            features[feature_name] = features_to_device(features[feature_name], device)
        else:
            features[feature_name] = features[feature_name].to(device)

    return features


def fullname(o):
    """
    Gives a full name (package_name.class_name) for a class / object in Python. Will
    be used to load the correct classes from JSON files
    """
    module = o.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return o.__class__.__name__  # Avoid reporting __builtin__
    else:
        return module + '.' + o.__class__.__name__


def import_from_string(dotted_path):
    """
    Import a dotted module path and return the attribute/class designated by the
    last name in the path. Raise ImportError if the import failed.
    """
    try:
        module_path, class_name = dotted_path.rsplit('.', 1)
    except ValueError:
        msg = "%s doesn't look like a module path" % dotted_path
        raise ImportError(msg)

    module = importlib.import_module(module_path)

    try:
        return getattr(module, class_name)
    except AttributeError:
        msg = 'Module "%s" does not define a "%s" attribute/class' % (module_path, class_name)
        raise ImportError(msg)
    
class Pooling(torch.nn.Module):
    """Performs pooling (max or mean) on the token embeddings."""
    def __init__(self,
                 word_embedding_dimension,
                 pooling_mode_cls_token=False,
                 pooling_mode_max_tokens=False,
                 pooling_mode_mean_tokens=True,
                 pooling_mode_mean_sqrt_len_tokens=False):
        super(Pooling, self).__init__()

        self.config_keys = ['word_embedding_dimension',  'pooling_mode_cls_token', 'pooling_mode_mean_tokens', 'pooling_mode_max_tokens', 'pooling_mode_mean_sqrt_len_tokens']

        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_mode_cls_token = pooling_mode_cls_token
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens
        self.pooling_mode_max_tokens = pooling_mode_max_tokens
        self.pooling_mode_mean_sqrt_len_tokens = pooling_mode_mean_sqrt_len_tokens

        pooling_mode_multiplier = sum([pooling_mode_cls_token, pooling_mode_max_tokens, pooling_mode_mean_tokens, pooling_mode_mean_sqrt_len_tokens])
        self.pooling_output_dimension = (pooling_mode_multiplier * word_embedding_dimension)

    def forward(self, features):
        token_embeddings = features['token_embeddings']
        cls_token = features['cls_token_embeddings']
        attention_mask = features['attention_mask']

        ## Pooling strategy
        output_vectors = []
        if self.pooling_mode_cls_token:
            output_vectors.append(cls_token)
        if self.pooling_mode_max_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            #If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if 'token_weights_sum' in features:
                sum_mask = features['token_weights_sum'].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if self.pooling_mode_mean_tokens:
                output_vectors.append(sum_embeddings / sum_mask)
            if self.pooling_mode_mean_sqrt_len_tokens:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))

        output_vector = torch.cat(output_vectors, 1)
        features.update({'sentence_embedding': output_vector})
        return features

    def get_sentence_embedding_dimension(self):
        return self.pooling_output_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        return Pooling(**config)
    
class Dense(torch.nn.Module):
    """Feed-forward function with  activiation function."""
    def __init__(self, 
                 in_features, 
                 out_features, 
                 bias=True, 
                 activation_function=torch.nn.Tanh()):
        super(Dense, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.activation_function = activation_function
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, features):
        features.update({'sentence_embedding': self.activation_function(self.linear(features['sentence_embedding']))})
        return features

    def get_sentence_embedding_dimension(self):
        return self.out_features

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump({'in_features': self.in_features, 'out_features': self.out_features, 'bias': self.bias, 'activation_function': fullname(self.activation_function)}, fOut)

        torch.save(self.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        config['activation_function'] = import_from_string(config['activation_function'])()
        model = Dense(**config)
        model.load_state_dict(torch.load(os.path.join(input_path, 'pytorch_model.bin'), map_location=torch.device('cpu')))
        return model
class EncoderModule(torch.nn.Module):
    def __init__(self, 
                 pretrained_model_name_or_path, 
                 pooling_model_name_or_path,
                 dense_model_name_or_path,
                 feature_extraction_mode='sentence_embedding'):
        super(EncoderModule, self).__init__()

        self.distilbert_config = transformers.AutoConfig.from_pretrained(pretrained_model_name_or_path)
        self.distilbert = transformers.AutoModel.from_pretrained(pretrained_model_name_or_path, config=self.distilbert_config)

        self.feature_extraction_mode = feature_extraction_mode
        self.is_pooling_module = False
        self.is_dense_module = False

        if os.path.exists(pooling_model_name_or_path):
            self.pooling_module = Pooling.load(pooling_model_name_or_path)
            self.is_pooling_module = True
        
        if os.path.exists(dense_model_name_or_path):
            self.dense_module = Dense.load(dense_model_name_or_path)
            self.is_dense_module = True
        
    
    def freeze_encoder(self):
        for param in self.distilbert.parameters():
            param.requires_grad = False
        
        for param in self.dense_module.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        for param in self.distilbert.parameters():
            param.requires_grad = True

        for param in self.dense_module.parameters():
            param.requires_grad = True

    def modules_forward(self, features):
        if not self.is_pooling_module:
            return features

        output_states = self.distilbert(**features)
        output_tokens = output_states[0]
        cls_tokens = output_tokens[:, 0, :]  # CLS token is first token

        features.update({
            'token_embeddings': output_tokens, 
            'cls_token_embeddings': cls_tokens, 
            'attention_mask': features['attention_mask']
        })
        if self.distilbert.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3:  # Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({'all_layer_embeddings': hidden_states})

        
        features = self.pooling_module.forward(features)
        
        if self.is_dense_module:
            features = self.dense_module.forward(features)

        return features
    
    def feature_extraction(self, features):
        out_features = self.modules_forward(features)
        if self.feature_extraction_mode not in out_features:
            return features

        embeddings = out_features[self.feature_extraction_mode]
        if self.feature_extraction_mode == 'token_embeddings':
            # Set token embeddings to 0 for padding tokens
            input_mask = out_features['attention_mask']
            input_mask_expanded = input_mask.unsqueeze(-1).expand(embeddings.size()).float()
            embeddings = embeddings * input_mask_expanded

        return embeddings

    def forward(self, features):
        # Feature extraction from Pre-Trained models
        embeddings = self.feature_extraction(features)

        # Do other stuff later
        # Do not forget this part ...

        return embeddings
    
pooling_module = Pooling.load(pooling_model_name)
#pooling_module.get_sentence_embedding_dimension()
dense_module = Dense.load(dense_model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = EncoderModule(
    pretrained_model_name_or_path=transformer_model_name,
    pooling_model_name_or_path=pooling_model_name,
    dense_model_name_or_path=dense_model_name,
)
encoder = encoder.to(device)
encoder.freeze_encoder()


for bi, d in tqdm(enumerate(sample_data_loader), total=len(sample_data_loader)):
    encoder_inputs = d['encoder_inputs']
    encoder_inputs = features_to_device(encoder_inputs, device)
    embeddings = encoder(encoder_inputs)
    if Gaussian_Noise :
      embeddings = add_Noise(embeddings.cpu())
    embeddings_by_handmade_model.append(embeddings)

    decoder_inputs = d['decoder_inputs']
    right_image_batch.append(decoder_inputs)


embeddings_by_handmade_model = torch.cat(embeddings_by_handmade_model, dim=0)
right_image_batch = torch.cat(right_image_batch, dim=0)
right_image_batch128 = torch.cat(right_image_batch128, dim=0)
right_image_batch256 = torch.cat(right_image_batch256, dim=0)
