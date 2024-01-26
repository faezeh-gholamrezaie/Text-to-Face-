
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
from model import *


class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.image_size = 64
        self.num_channels = 3
        self.noise_dim = 100
        #self.embed_dim = 1024
        self.embed_dim = 512
        self.projected_embed_dim = 128
        self.latent_dim = self.noise_dim + self.projected_embed_dim
        self.ngf = 64
        #state siz = 100


        self.projection = nn.Sequential(
            nn.Linear(in_features=self.embed_dim, out_features=self.projected_embed_dim),
            nn.BatchNorm1d(num_features=self.projected_embed_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

        self.netG = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2,self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(self.ngf, self.num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
             # state size. (num_channels) x 64 x 64
            )
        self.netG2 = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, self.ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 16, self.ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 8,self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(self.ngf * 2,self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(self.ngf, self.num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
             # state size. (num_channels) x 128 x 128
            )
        
        self.netG3 = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, self.ngf * 32, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 32),
            nn.ReLU(True),
            # state size. (ngf*32) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 32, self.ngf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 16, self.ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 8,self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 32 x 32
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 64 x 64
            nn.ConvTranspose2d(self.ngf * 2,self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 128 x 128
            nn.ConvTranspose2d(self.ngf, self.num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
             # state size. (num_channels) x 256 x 256
            )


    def forward(self, embed_vector, z ,number_net):

        projected_embed = self.projection(embed_vector).unsqueeze(2).unsqueeze(3) #[100, 128]
        latent_vector = torch.cat([projected_embed, z], 1) #[100, 228, 1, 1]
        #atent_vector = torch.cat([projected_embed, z], 0)
        #latent_vector = embed_vector
        if number_net == 1:
          output = self.netG(latent_vector) #[100, 3, 64, 64]
        elif number_net == 2:
          output = self.netG2(latent_vector) #[100, 3, 128, 128]
        else:
          output = self.netG3(latent_vector) #[100, 3, 256, 256]
        
        return output
    
#This discriminator captures image_wrong and text_real at the input
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.image_size = 64
        self.num_channels = 3
        #self.embed_dim = 1024
        self.embed_dim = 512
        self.projected_embed_dim = 128
        self.ndf = 64
        self.ndf2 = 32
        self.ndf3 = 16
        self.B_dim = 128
        self.C_dim = 16

        self.netD_1 = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.num_channels, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4,self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.netD_1_in128 = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(self.num_channels, self.ndf2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf2) x 64 x 64
            nn.Conv2d(self.ndf2, self.ndf2 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf2 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf2*2) x 32 x 32
            nn.Conv2d(self.ndf2 * 2, self.ndf2 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf2 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf2*2) x 16 x 16
            nn.Conv2d(self.ndf2 * 4,self.ndf2 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf2 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf2*4) x 8 x 8
            nn.Conv2d(self.ndf2 * 8,self.ndf2 * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf2 * 16),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.netD_1_in256 = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(self.num_channels, self.ndf3, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf3) x 128 x 128
            nn.Conv2d(self.ndf3, self.ndf3 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf3 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf3*2) x 64 x 64
            nn.Conv2d(self.ndf3 * 2, self.ndf3 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf3 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf3*2) x 32 x 32
            nn.Conv2d(self.ndf3 * 4,self.ndf3 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf3 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf3*4) x 16 x 16
            nn.Conv2d(self.ndf3 * 8,self.ndf3 * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf3 * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf3*8) x 8 x 8
            nn.Conv2d(self.ndf3 * 16,self.ndf3 * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf3 * 32),
            nn.LeakyReLU(0.2, inplace=True)
        )

        #self.projector = utils.Concat_embed(self.embed_dim, self.projected_embed_dim)
        self.projector = Concat_embed(self.embed_dim, self.projected_embed_dim)

        self.netD_2 = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8 + self.projected_embed_dim, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            )

    def forward(self, inp, embed,number_netD):
        if number_netD == 1:
          x_intermediate = self.netD_1(inp) #[100, 512, 4, 4]

        elif number_netD == 2:
          x_intermediate = self.netD_1_in128(inp) #[100, 512, 4, 4]

        else:
          x_intermediate = self.netD_1_in256(inp) #[100, 512, 4, 4]

        x = self.projector(x_intermediate, embed) #[100, 640, 4, 4]        
        x = self.netD_2(x) #[100, 1, 1, 1]


        return x.view(-1, 1).squeeze(1) , x_intermediate