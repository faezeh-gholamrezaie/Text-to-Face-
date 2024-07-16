# Text to Face

A Novel Encoder-Decoder Approach for Text-to-Face Conversion using Generative Adversarial Networks and Attention Mechanism

This repository contains a TensorFlow implementation for generating face images using the GAN-CLS Algorithm, as described in the paper [Generative Adversarial Text-to-Image Synthesis][1](https://proceedings.mlr.press/v48/reed16.html). The model is built upon the solid foundation of the [DCGAN in TensorFlow][2](https://github.com/tensorlayer/DCGAN).

With the ability to handle input descriptions in over 100 languages, this implementation is highly versatile and adaptable. This means you can generate face images based on textual descriptions in various languages.

To train and evaluate the model, we utilize the large-scale CelebFaces Attributes (CelebA) dataset, which provides a diverse set of realistic human faces.
<p align="center">
<img src="./images/Picture1.jpg" width="500" height="200">
</p>
Image Source : [https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip]

Caption source : [https://raw.githubusercontent.com/midas-research/text2facegan/master/data/caps.txt]

## Results

-The man sports a 5 o’clock shadow.He has big nose.The young attractive man is smiling.

<p align="center">
<img src="./images/Picture4.png" width="100" height="100">
</p>

##text-to-face evaluation (human evaluation) : 

[Text2FaceGAN][3](https://ieeexplore.ieee.org/abstract/document/8919389).

<p align="center">
<img src="./images/Picture2.jpg" width="400" height="200">
</p>

## text-to-face evaluation (Frechet Inception Distance (FID)) :

|      Model     |      FID   |
| -------------- | ---------- |
| Cycle Text2Face| 1.20±0.081 |
| Text2FaceGan   | 1.4±0.7    |
# References

Article cycle text2face: cycle text-to-face gan via transformers : [https://arxiv.org/abs/2206.04503]

