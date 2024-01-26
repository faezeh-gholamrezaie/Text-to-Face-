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
from data_loader import  *
class EncoderDecoderDataset(torch.utils.data.Dataset):
    def __init__(self, texts, images, tokenizer,max_size=64 ,max_length=128, padding='max_length', truncation='only_first'):
        assert len(texts) == len(images)

        self.texts = texts
        self.images = images
        self.tokenizer = tokenizer

        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

        #images Decoder Information   
        #,transforms.RandomHorizontalFlip()

        self.transform =transforms.Compose([
        transforms.Scale(int(max_size)), 
        transforms.RandomCrop(max_size)])

        self.normalize = transforms.Compose([
        transforms.ToTensor()])

        

    def __getitem__(self, item):
        text = str(self.texts[item])
        text = ' '.join(text.split())
        image = self.images[item]

        _encoder_inputs = self.tokenizer(
            text,
            padding=self.padding,
            max_length=self.max_length,
            truncation=self.truncation,
            # return_tensors='pt'
        )
        encoder_inputs = {
            'input_ids': torch.tensor(_encoder_inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(_encoder_inputs['attention_mask'], dtype=torch.long),
        }
        
        # New decoder
        transform=self.transform
        normalize=self.normalize

        _decoder_inputs = Image.open(image).convert('RGB')  
        decoder_inputs = self.normalize(self.transform(_decoder_inputs))

        return {
            'encoder_inputs': encoder_inputs, 
            'decoder_inputs': decoder_inputs,
        }


    def __len__(self):
        return len(self.texts)
    
transformer_model_name = SBERT_MODEL_NAME_OR_PATH + '/transformer/'
pooling_model_name = SBERT_MODEL_NAME_OR_PATH + '/pooling/'
dense_model_name = SBERT_MODEL_NAME_OR_PATH + '/dense/'

tokenizer = transformers.AutoTokenizer.from_pretrained(transformer_model_name, use_fast=True)
sample_data = data.iloc[:].values
sample_dataset = EncoderDecoderDataset(
    texts=sample_data[:, 0], 
    images=sample_data[:, 1], 
    tokenizer=tokenizer, 
    max_length=128,
    )