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

dataset = 'CelebA' 
DATASET_PATH = ""  #Path image and caption(caps.txt)

if dataset == 'CelebA':
    """
    images.shape = [192010 , 64, 64, 3]
    captions_ids = [192010 , any]
    """
    data = pd.read_csv(DATASET_PATH + '/caps.txt', sep="\t", names=['img_path', 'desc'])
    data['desc'] = data['desc'].apply(lambda t: t if isinstance(t, str) else None)
    data = data.dropna()
    data['desc'] = data['desc'].apply(lambda t: t if len(t) > 16 else None)
    data = data.dropna()
    data = data.drop_duplicates()
    data['desc'] = data['desc'].apply(lambda t: t.replace('|', ' '))
    data['img_path'] = data['img_path'].apply(lambda t: os.path.join(DATASET_PATH, 'img_align_celeba', t))
    data = data[['desc', 'img_path']]
    data = data.astype('str')
    data = data.reset_index(drop=True)
    data.head()
