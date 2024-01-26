
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
from Gan_2 import *


class Utils(object):

    @staticmethod
    def smooth_label(tensor, offset):
        return tensor + offset

    @staticmethod
    def save_checkpoint(netD, netG, netG2, dir_path, subdir_path, epoch):
        path =  os.path.join(dir_path, subdir_path)
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(netD.state_dict(), '{0}/disc_{1}.pth'.format(path, epoch))
        torch.save(netG.state_dict(), '{0}/gen_{1}.pth'.format(path, epoch))
        torch.save(netG2.state_dict(), '{0}/gen2_{1}.pth'.format(path, epoch))


    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


class Trainer(object):
    def __init__(self, type, dataset, split, lr, 
                 save_path, l1_coef, l2_coef, pre_trained_gen, pre_trained_disc, batch_size, num_workers, epochs):
        #with open('config.yaml', 'r') as f:
        #    config = yaml.load(f)

        self.generator = torch.nn.DataParallel(generator().cuda())
        self.discriminator = torch.nn.DataParallel(discriminator().cuda())

        EMBEDDING_DIM=512
        HIDDEN_DIM=256
        self.generator_2 = torch.nn.DataParallel(BERT_CNN_ENCODER_RNN_DECODER(EMBEDDING_DIM, HIDDEN_DIM,
                                             rec_unit='LSTM').cuda())

        if pre_trained_disc:
            self.discriminator.load_state_dict(torch.load(pre_trained_disc))

        else:
            self.discriminator.apply(Utils.weights_init)

        if pre_trained_gen:
            self.generator.load_state_dict(torch.load(pre_trained_gen))
        else:
            self.generator.apply(Utils.weights_init)

        #if dataset == 'birds':
        #    self.dataset = Text2ImageDataset(config['birds_dataset_path'], split=split)
        #elif dataset == 'flowers':
        #    self.dataset = Text2ImageDataset(config['flowers_dataset_path'], split=split)
        #else:
        #    print('Dataset not supported, please select either birds or flowers.')
        #    exit()
        
        #print "Image = ",len(self.dataset)
        self.noise_dim = 100
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.beta1 = 0.9
        self.num_epochs = epochs


        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.l2_coef_g2 = l2_coef

        #self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
        #                        num_workers=self.num_workers)

        self.optimD = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimG = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        self.optimG_2 = torch.optim.Adam(self.generator_2.parameters(), lr=self.lr, betas=(self.beta1, 0.999))


        self.logger = Logger()
        self.checkpoints_path = 'checkpoints'
        self.save_path = save_path
        self.type = type

    def train(self, pretrain,iteration):

        if self.type == 'gan':
            self._train_gan(pretrain,iteration)

    def load_checkpoint(self, netD_path, netG_path, netG2_path):

        device = torch.device("cuda")

        self.discriminator.load_state_dict(torch.load(netD_path))
        self.discriminator.to(device)

        self.generator.load_state_dict(torch.load(netG_path))
        self.generator.to(device)

        self.generator_2.load_state_dict(torch.load(netG2_path))
        self.generator_2.to(device)


    def _train_gan(self, pretrain, iteration):
        criterion = nn.BCELoss() #text , image
        l2_loss = nn.MSELoss() #text , text
        l1_loss = nn.L1Loss() #image , image

        
        pri_g_loss = 0
        
        for epoch in range(self.num_epochs):

                
                right_embed = embeddings_by_handmade_model.cuda()

                right_images = right_image_batch
                number_image = 1

                real_labels = torch.ones(right_images.size(0))
                fake_labels = torch.zeros(right_images.size(0))

                smoothed_real_labels = torch.FloatTensor(Utils.smooth_label(real_labels.numpy(), 0))

                real_labels = Variable(real_labels).cuda()
                smoothed_real_labels = Variable(smoothed_real_labels).cuda()
                fake_labels = Variable(fake_labels).cuda()

                #Train the generator_2
                self.generator_2.zero_grad()
                self.embed_text = right_embed.reshape(1,100,512)
                output_token_text, words_features, sent_code, word_logits = self.generator_2(right_images, self.embed_text)
                real_loss_g2 = l2_loss(word_logits.reshape(100,512), self.embed_text )

                real_loss_g2.backward()
                self.optimG_2.step()

                # Train the discriminator
                self.discriminator.zero_grad()
                outputs, activation_real = self.discriminator(right_images, right_embed, number_image)
                real_loss = criterion(outputs, smoothed_real_labels)
                real_score = outputs

                m = torch.distributions.Normal(0,  1)
                noise = m.sample((100,100,1,1))
                fake_images = self.generator(right_embed, noise, number_image)
                outputs, _  = self.discriminator(fake_images, right_embed, number_image)
                fake_loss   = criterion(outputs, fake_labels)
                fake_score  = outputs

                d_loss = real_loss + fake_loss

                #if cls:
                #    d_loss = d_loss + wrong_loss

                d_loss.backward()
                self.optimD.step()
              
                # Train the generator
                self.generator.zero_grad()
                fake_images = self.generator(right_embed, noise, number_image)
                outputs, activation_fake = self.discriminator(fake_images, right_embed, number_image)
                _, activation_real = self.discriminator(right_images, right_embed, number_image)

                activation_fake = torch.mean(activation_fake, 0)    #try with median and check if it converges
                activation_real = torch.mean(activation_real, 0)    #try with median and check if it converges

                #generator_2
                output_token_text, words_features, sent_code, word_logits = self.generator_2(fake_images, self.embed_text)
                real_loss_g2 = self.l2_coef_g2 *(l2_loss(word_logits.reshape(100,512), self.embed_text )) 
                #


                g_loss =  real_loss_g2 + criterion(outputs.cuda(), real_labels.cuda())+ self.l2_coef * l2_loss(activation_fake.cuda(), activation_real.cuda().detach())+ self.l1_coef * l1_loss(fake_images.cuda(), right_images.cuda())
                
                g_loss.backward()
                self.optimG.step()
                self.optimG_2.step()
                
                #print('iter:', iteration)

                if (epoch+1) % 10 == 0:
                    self.logger.log_iteration_gan(epoch, iteration, d_loss, g_loss, real_score, fake_score)
                    print("G2 :", real_loss_g2)

                if g_loss  < 0.8 :
                   Utils.save_checkpoint(self.discriminator, self.generator, self.generator_2 ,'/content/gdrive/MyDrive/Colab Notebooks/Text2Face', self.save_path, iteration)
                   self.logger.log_iteration_gan(epoch, iteration, d_loss, g_loss, real_score, fake_score)
                   print("G2 :", real_loss_g2)
                   print("g_loss : <0.8")
                   if pretrain :
                     break

                   #Utils.save_checkpoint(self.discriminator, self.generator, self.generator_2 ,self.checkpoints_path, self.save_path, epoch)
                if g_loss < 0.6 :
                   Utils.save_checkpoint(self.discriminator, self.generator, self.generator_2 ,'/content/gdrive/MyDrive/Colab Notebooks/Text2Face', self.save_path, iteration)
                   #Utils.save_checkpoint(self.discriminator, self.generator, self.generator_2 ,self.checkpoints_path, self.save_path, epoch)
                   self.logger.log_iteration_gan(epoch, iteration, d_loss, g_loss, real_score, fake_score)
                   print("G2 :", real_loss_g2)
                   print("g_loss : <0.6")
                   break           

                if epoch == self.num_epochs-1 :
                   Utils.save_checkpoint(self.discriminator, self.generator, self.generator_2 ,'/content/gdrive/MyDrive/Colab Notebooks/Text2Face', self.save_path, iteration)
                   #Utils.save_checkpoint(self.discriminator, self.generator, self.generator_2 ,self.checkpoints_path, self.save_path, epoch)
                   self.logger.log_iteration_gan(epoch, iteration, d_loss, g_loss, real_score, fake_score)
                   print("G2 :", real_loss_g2)
                   print("g_loss : finish")
                   break
                iteration += 1
                pri_g_loss = g_loss


    def predict(self):
            
            right_images = right_image_batch
            number_image = 1
            right_embed = embeddings_by_handmade_model.cpu()
            txt = sample_data[:, 0]

            if not os.path.exists('results/{0}'.format(self.save_path)):
                os.makedirs('results/{0}'.format(self.save_path))

            m = torch.distributions.Normal(0,  1)
            noise = m.sample((100,100,1,1))
            fake_images = self.generator(right_embed, noise, number_image)
            fake_image_batch.append(fake_images)

            for image, t in zip(fake_images, txt):
                im = Image.fromarray(image.data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
                im.save('results/{0}/{1}.jpg'.format(self.save_path, t.replace("/", "")[:100]))
                #image1 = T.ToPILImage(mode='RGB')(image1)
                display_markdown('Reconstructed image:')
                display(im)
                print(t)


args = easydict.EasyDict({'type': 'gan', 
                         'lr': 0.0002,
                         'l1_coef': 50,
                         'l2_coef': 100,
                         'pretrain': True,
                         'save_path':'./CelebA_test',

'inference': True,
'pre_trained_disc': 'checkpoints/flowers_cls/disc_190.pth',
'pre_trained_gen': 'checkpoints/flowers_cls/gen_190.pth',
'dataset': 'flowers', 
'split': 2,
'batch_size':64,
'num_workers':8,
'epochs':20000})

trainer = Trainer(type=args.type,
                  dataset=args.dataset,
                  split=args.split,
                  lr=args.lr,
                  save_path=args.save_path,
                  l1_coef=args.l1_coef,
                  l2_coef=args.l2_coef,
                  pre_trained_disc=False,
                  pre_trained_gen=False,
                  batch_size=args.batch_size,
                  num_workers=args.num_workers,
                  epochs=args.epochs
                  )
trainer.predict()