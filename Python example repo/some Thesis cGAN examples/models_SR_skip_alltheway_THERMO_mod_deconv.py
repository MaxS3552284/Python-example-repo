# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 19:54:37 2022

Superresolution Network by Maximilian Schmitt 3552284
based on the Pix2Pix Network by Isola et al.

Model for 2x Superresolution, with LR image input

"""

import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
from torchvision.transforms import Resize
import torch.nn.functional as F
import matplotlib.pyplot as plt


LOCATION = 'HTW'
# LOCATION = 'HOME'

if LOCATION =='HTW':
    directory = "C:/Users/vc-lab/Desktop/Max_Thesis_Python_and_Stuff/super_resolution_project/BSDS500_train"
    save_location = "C:/Users/vc-lab/Desktop/Max_Thesis_Python_and_Stuff/super_resolution_project/saved_models/"
elif LOCATION =='HOME':
    directory = "C:/Users/Max/Desktop/HTW Studium/BMT Master/Master Thesis/super_resolution_project/BSDS500_train"
    save_location = "C:/Users/Max/Desktop/HTW Studium/BMT Master/Master Thesis/super_resolution_project/saved_models/"

def set_device():
    # check if cuda device is there, if not load model and data on the cpu instead
    if torch.cuda.is_available():
        dev = "cuda:0"
        print(dev)
    else:
        dev = "cpu"
        print(dev)
    return torch.device(dev)
DEVICE = set_device()


def _weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)) and m.requires_grad_():
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d) and m.requires_grad_():
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

def De_NormalizeData(data): # reverse normalization between 0 and 1
    # 45, 17 are the max and min values over a dataset
    # 0 might be lowest value due to cut out area 
    return (data * (45 - 17) + 17)

def display_progress(ground_truth, cond, fake, figsize=(40, 20)):
    ground_truth = ground_truth.detach().cpu()
    cond = cond.detach().cpu()#.permute(1, 2, 0)
    fake = fake.detach().cpu()#.permute(1, 2, 0)
    cond_up_trans = Resize((128*2,128*2), interpolation=Image.BICUBIC)
    cond = cond_up_trans(cond)

    fig, ax = plt.subplots(1, 2, figsize=figsize)

    ground_truth_img = De_NormalizeData(np.swapaxes(np.asarray(ground_truth.T),0,1))
    cond_img = De_NormalizeData(np.swapaxes(np.asarray(cond.T),0,1))
    fake_img = De_NormalizeData(np.swapaxes(np.asarray(fake.T),0,1))

    im1=ax[0].imshow(ground_truth_img, cmap='gray')
    im2=ax[1].imshow(fake_img, cmap='gray')
    ax[0].title.set_text('Ground Truth')
    ax[1].title.set_text('Generated')
    plt.colorbar(im1,ax=ax[0])
    plt.colorbar(im2,ax=ax[1])
    plt.show()
    
    fig2, ax2 = plt.subplots(1, 2, figsize=figsize)
    im3=ax2[0].imshow(cond_img)
    im4=ax2[1].imshow(fake_img)
    ax2[0].title.set_text('Input')
    ax2[1].title.set_text('Generated')
    plt.colorbar(im3,ax=ax2[0])
    plt.colorbar(im4,ax=ax2[1])
    plt.show()



class DownSampleConv(nn.Module):
    # applies 2-dimensional convolutional layer 
    # also applies batchnorm and activation(leaky ReLU) layer if argument is true
    def __init__(self, in_channels, out_channels, kernel=4, strides=2, padding=1, activation=True, batchnorm=True):
        """
        Paper details:
        - C64-C128-C256-C512-C512-C512-C512-C512
        - All convolutions are 4×4 spatial filters applied with stride 2
        - Convolutions in the encoder downsample by a factor of 2
        """
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm

        self.conv = nn.Conv2d(in_channels, out_channels, kernel, strides, padding, padding_mode='reflect')

        if batchnorm: # additionally apply BN if true
            self.bn = nn.BatchNorm2d(out_channels)

        if activation: # additionally apply activation layer if true
            self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        if self.batchnorm:
            x = self.bn(x)
        if self.activation:
            x = self.act(x)
        return x


class UpSampleConv(nn.Module):
    # applies 2-dimensional transposed convolutional layer(subpixel conv)
    # also applies batchnorm-, activation(ReLU) layer and dropout if arguments is true
    def __init__(
        self, in_channels, out_channels, kernel=4, strides=2, padding=1, activation=True, batchnorm=True, dropout=False
    ):
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm
        self.dropout = dropout

        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel, strides, padding)

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)

        if activation:
            self.act = nn.ReLU(True)

        if dropout:
            # dropout in generator serves to apply gaussian noise (z) as conditioning 
            # only adding small stochasticity, more would be ideal, but better than none
            # gaussian noise added as input to the generator will only lead to the generator learning to ignore it
            self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.deconv(x)
        if self.batchnorm:
            x = self.bn(x)

        if self.dropout:
            x = self.drop(x)
        return x


class Generator(nn.Module):
    # G:{x,z,yHD}   x = input image, z = conditioning, yHD = ground truth(HD image)
    def __init__(self, in_channels, out_channels):
        """
        Paper details:
        - Encoder: C64-C128-C256-C512-C512-C512-C512-C512
        - All convolutions are 4×4 spatial filters applied with stride 2
        - Convolutions in the encoder downsample by a factor of 2
        - Decoder: CD512-CD1024-CD1024-C1024-C1024-C512 -C256-C128
        """
        super().__init__()
        
        # encoder/donwsample convs
        # input dimensions: bs x 1 x 128 x 128
        self.encoders = [
            DownSampleConv(in_channels, 128, batchnorm=False),  # output dimensions of layer: bs x 128 x 64 x 64 
            DownSampleConv(128, 256),  # bs x 256 x 32 x 32 
            DownSampleConv(256, 512),  # bs x 512 x 16 x 16
            DownSampleConv(512, 512),  # bs x 512 x 8 x 8
            DownSampleConv(512, 512),  # bs x 512 x 4 x 4
            DownSampleConv(512, 512),  # bs x 512 x 2 x 2
            DownSampleConv(512, 512, batchnorm=False),  # bs x 512 x 1 x 1
        ]# C128-C256-C512-C512-C512-C512-C512

        # U-NET structure is mirrored, thats why input and output channel of layer 6 of the encoder
        # are equal the output and input channel of layer 1 of the decoder,
        # and the decoder input channel of layer 2-6 being double the number of output channel in the encoder layer 5-1,
        # since they are directly connected with skip connections
        # decoder/upsample convs
        self.decoders = [
            UpSampleConv(512, 512, dropout=True),  # bs x 512 x 2 x 2
            UpSampleConv(1024, 512, dropout=True),  # bs x 512 x 4 x 4 # doppelte input channel weil skipp connection
            UpSampleConv(1024, 512, dropout=True),  # bs x 512 x 8 x 8
            UpSampleConv(1024, 512),  # bs x 512 x 16 x 16
            UpSampleConv(1024, 256),  # bs x 256 x 32 x 32
            UpSampleConv(512, 128),  # bs x 128 x 64 x 64
            # MUST NOT HAVE MORE THAN 6 LAYERS, OR INDEX FURTHER DOWN NEEDS TO BE CHANGED
            # otherwise skipconnections dont link correctly up with the encoder anymore !!!
            # and python starts complaining about missmatching channel numbers

        ] # 512-1024-1024-1024-1024-512-128
        # self.decoder_channels = [512, 512, 512, 512, 512, 256, 128, 64, 64] # <- -decoder input channel?
        
        # self.up_samp_input_goes_here = [
        #     DownSampleConv(256, 256,strides=1,kernel=1),
        #     DownSampleConv(256, 256,strides=1,kernel=3),
        # ]
        # add new upsample layers here, for super ressolution
        self.sup_res_layer = [
            UpSampleConv(128, 128), # bs x 128 x 128 x 128
        ]

        self.final_conv = nn.ConvTranspose2d(128, out_channels, kernel_size=3, stride=1, padding=1)#64 previously
        self.final_conv_new = [
            UpSampleConv(128, 128,kernel=4, strides=2), # bs x 128 x 128 x 128
            UpSampleConv(128, 128,kernel=3, strides=1),
            UpSampleConv(128, 128,kernel=3, strides=1),
            UpSampleConv(128, 128,kernel=3, strides=1)
        ]
        
        # bs x 1 x 128 x 128
        self.tanh = nn.Tanh()

        # self.upsamp_input = nn.ModuleList(self.up_samp_input_goes_here)
        self.encoders = nn.ModuleList(self.encoders)
        self.sup_resser = nn.ModuleList(self.sup_res_layer)
        self.decoders = nn.ModuleList(self.decoders)
        self.new_final_conv = nn.ModuleList(self.final_conv_new)
    
    # test network for step, generate image
    def forward(self, x, upscaled_input):
        # print("low res data size change u-net encoder after: ", x.size())
        skips_cons = [] # empty list for skip connections
        # y_up_input = self.upsamp_input[0](upscaled_input) 
        for encoder in self.encoders:
            # applyies encoder layer one after another on input data x
            
            x = encoder(x) # 
            # print("low res data size change u-net encoder after: ", x.size())

            skips_cons.append(x) # append outut of encoder to corresponding skip connection
            # results in list 'skips_cons' filled with encoder outputs
        # get a list of the indexes of the skip cons, except the last, and reverses the indexes 
        # for matching the connections of the encoder with the reversed indices of the decoder layers 
        skips_cons = list(reversed(skips_cons[:-1])) # [:-1] = all elements of the sequence but the last
        decoders = self.decoders[:-1] # all but last decoder layer
        # wich means, dont add more layers to decoder list or change this index

        for decoder, skip in zip(decoders, skips_cons): 
            # zip matches the iterables of the two given lists within a tuple and returns them as a combined list
            # resulting in decoder layers and skip connections being matched
            # for ervery 'decoder'-index/object in 'zip(content)'-list 'skip = zip(content)[decoder]'
            x = decoder(x) # apply decoder layer
            # print("low res data size change u-net decoder after: ", x.size())
            m, n = skip.shape[-2:] # get shape of skip connection from the last two items in list
            x = torch.cat((x, skip), axis=1) # concatenates/connects tensors. tensors must be same shape on axis 1
            # in this case x = decoder output, skip = list containing corresponding encoder output, 
            # decoder = corresponding decoder layer
            # resulting in the compination of x with its corresponding encoder output

        x = self.decoders[-1](x)    # applies last decoder layer
        # print("low res data size change u-net decoder after: ", x.size())
        x = self.sup_resser[0](x)   # apply super ressolution layer
        # print("low res data size change u-net decoder after: ", x.size())
        # print("x after upscaleing: ", x.size())
        # print("upscaled input: ", upscaled_input.size())
        x = self.new_final_conv[0](x)
        x = x + upscaled_input
        # print("x after old output: ", x.size())
        # SHARPENING FILTER CONSISTS OF THE FOLLOWING 4 LAYERS
        x = self.new_final_conv[1](x)
        # print("x after 1. new: ", x.size())
        x = self.new_final_conv[2](x)
        # print("x after 2. new: ", x.size())
        x = self.new_final_conv[3](x)
        # print("x after 3. new: ", x.size())
        # print("x after second new layer: ", x.size())
        x = self.final_conv(x)      # apply output layer
        # print("x after final conv: ", x.size())
        # for up_samp_input_goes_here in self.upsamp_input:
        #     x = up_samp_input_goes_here(x)
        #     print("apply deconf: ", x.size())
        
        # x = torch.cat((x, upscaled_input), axis=1) # combine input_up and last layer input
        # print("x + y_up: ", x.size())
        return self.tanh(x)         # tanh? why?


class PatchGAN(nn.Module):
    # Discriminator
    # D:{G(x,z) or xLable , yUS}  G(x,z) = generator output, xLable = ground truth(lable, whether real or fake), yUS = upscaled(bicubic) input image
    # in original Pix2Pix yUS would be equivalent to the input image x and ground truth
    # note that using yHD instead of yUS might cause the discriminator to lable anything with error >0 as fake,
    # since it would compare real images with themself
    def __init__(self, input_channels):
        super().__init__()
        self.d1 = DownSampleConv(input_channels, 64, batchnorm=False)# 64 previously
        self.d2 = DownSampleConv(64, 128)# 64, 128 previously
        self.d3 = DownSampleConv(128, 256)
        self.d4 = DownSampleConv(256, 512)
        self.d5 = DownSampleConv(512, 512)
        self.final = nn.Conv2d(512, 1, kernel_size=1)
        
    # test network for step and calculate loss between ground truth and generated image
    def forward(self, x, y):
        # x = generator output or ground truth| y = upscaled low res input/condition data image
        # in original Pix2Pix the size of the y tensor indicates it to be the low res data input x, which made sense, 
        # since the in the original Pix2Pix (which was not for SR but image manipulation), the input data doubled as ground truth 
        # since it also contained information about what constitutes for real images 
        # which would still fit, since in the original code x = yHD, but we upscale x and have a seperate HD ground truth
        # for example the image for a shoe, while the output color didn't matter the key characteristics were also deliverd by the input image
        # this does not wor with SR, sincewhat makes the image real, the "sharpness" musst not be containt in the input,
        # otherwise there is no point to the algorithm
              
        # compare low res data with high res 
        # x_up_trans = torch.nn.Upsample(scale_factor=2, mode='bicubic')
        # if x.shape != y.shape:
        #     x = x_up_trans(x)
        # else:
        #     x = x

        x = torch.cat([x, y], axis=1)
        x0 = self.d1(x)
        x1 = self.d2(x0)
        x2 = self.d3(x1)
        x3 = self.d4(x2)
        x4 = self.d5(x3)
        xn = self.final(x4)
        return xn


class Pix2Pix(nn.Module):
    # the cGAN model
    
    def __init__(self, in_channels=1, out_channels=1, learning_rate=0.0002, lambda_recon=200, display_step=25,
                 model_names=None):
        # defaults for learning rate, from pix2pix paper
        super().__init__()

        self.display_step = display_step
        self.gen = Generator(in_channels, out_channels)         # generator, U-Net
        self.patch_gan = PatchGAN(in_channels + out_channels)   # discriminator PatchGAN PatchGAN(4)
        self.lambda_recon = lambda_recon # patchgan reconstruction weight| balance between artifacts and smoothness
        self.display_step = display_step 
        self.learning_rate = learning_rate
        self.y_up_trans = torch.nn.Upsample(scale_factor=2, mode='bicubic')

        # intializing weights
        if  model_names is not None and os.path.isfile(save_location + model_names[0]):
            print('generator model file exists and is to be loaded')
            self.gen = torch.load(save_location + model_names[0])
        if model_names is not None and os.path.isfile(save_location + model_names[1]):
            print('discriminator model file exists and is to be loaded')
            self.patch_gan = torch.load(save_location  +model_names[1])
        else:
            self.gen = self.gen.apply(_weights_init)


        # initialize loss types used
        self.adversarial_criterion = nn.BCEWithLogitsLoss() # adverserial loss criterion
        self.recon_criterion = nn.L1Loss()                  # reconstruction loss criterion
    
        self.learner = [self.gen, self.patch_gan]

        for l in self.learner:
            l.to(DEVICE)

    def compile(self, optimizer="adam", learning_rate=3e-4):
        optimizer_fn = torch.optim.Adam # self._get_optimizer(optimizer)
        self.gen_optimizer = optimizer_fn(self.learner[0].parameters(), lr=learning_rate)
        self.disc_optimizer = optimizer_fn(self.learner[1].parameters(), lr=learning_rate)
        self.optimizer = optimizer_fn(self.learner[1].parameters(), lr=learning_rate)

        #self.disc_optimizer = self.prepare_optimizer(self.disc_optimizer)
        #self.gen_optimizer = self.prepare_optimizer(self.gen_optimizer)

        self._compiled = True

    def _gen_step(self, conditioned_images, rgb_images):
        # conditioned_images = generator output
        # rgb_images = Ground truth HD images
        # Pix2Pix has adversarial and a reconstruction loss
        # First calculate the adversarial loss
        conditioned_images_UP = self.y_up_trans(conditioned_images)
        # calculate generator loss during training step
        
        # fake images provided by generator class
        fake_images = self.gen(conditioned_images, conditioned_images_UP)

        disc_logits = self.patch_gan(fake_images, conditioned_images_UP)
        # disc_logits = self.patch_gan(fake_images, rgb_images)
        adversarial_loss = self.adversarial_criterion(disc_logits, torch.ones_like(disc_logits))

        recon_loss = self.recon_criterion(fake_images, rgb_images) # reconstruction loss
        lambda_recon = self.lambda_recon

        return adversarial_loss + lambda_recon * recon_loss
        # return recon_loss

    def _disc_step(self, conditioned_images, real_images):
        # conditioned_images = generator output or bicubic upscaled real images,
        # real_images = should be real hd images, 
        # Ground truth as conditioning would lead to an error of zero... if that is the case,
        # wouldn't the network learn to classify everything with error >0 to be fake?
        conditioned_images_UP = self.y_up_trans(conditioned_images)
        
        # backprob mini batch step: calculate discriminator loss during training step for minibatch 
        fake_images = self.gen(conditioned_images, conditioned_images_UP).detach() # gets fake images from generator
        fake_logits = self.patch_gan(fake_images, conditioned_images_UP) # feed discriminator with fake and condition images
        # fake_logits = self.patch_gan(fake_images, real_images)
        real_logits = self.patch_gan(real_images, conditioned_images_UP) # feed discriminator with ground truth images and condition imanges
        # real_logits = self.patch_gan(conditioned_images, real_images)
        # apply adversarial loss criterion on logits, and tensors of same size, filled with zeros or ones, 
        # depending on wether the images are real or fake
        # This is used for measuring the error of a reconstruction in for example an auto-encoder. 
        # Note that the targets t[i] should be numbers between 0 and 1.
        fake_loss = self.adversarial_criterion(fake_logits, torch.zeros_like(fake_logits))
        real_loss = self.adversarial_criterion(real_logits, torch.ones_like(real_logits))
        # adversarial_criterion = cross entropy loss
        # torch.zeros_like() as example for bad prediction
        # torch.ones_like() as example for good prediction
        # how far is fake off of the bad example and how far is real off of the good example?
        return (real_loss + fake_loss) / 2

    def train_step(self, batch, display=False): # manages the training:
        # take as input the data of the batch and the boolean value
        
        condition, real = batch
        conditioned_images_UP = self.y_up_trans(condition)
        
        # condition = batch[0] = data[0] = conditioning images, input, LR resolution
        # real = batch[1] = data[1] = real images, ground truth, HR resolution
        # print("input batch: batch[0] = condition, should be size of LR image ", condition.size())
        # print("input batch: batch[1] = real/ground truth, should be size of HD image ", real.size())
        
        # DISCRIMINATOR TRAINIMG:
        self.gen_optimizer.zero_grad() # Sets the gradients of all optimized torch.Tensors to zero
        self.disc_optimizer.zero_grad()
        
        # discriminator is at self.learner[1]
        disc_loss = self._disc_step(condition, real)
        #self.tracker.track_loss("discriminator/loss", disc_loss)
        # self.backward(disc_loss)
        disc_loss.backward() # calculates gradient via backwards propagation
        self.disc_optimizer.step() # optimizer.step is performs a parameter update based on the current gradient (stored in .grad attribute of a parameter) and the update rule. 

        # GENERATOR TRAINING:
        # generator is at self.learner[0]
        self.gen_optimizer.zero_grad()  # set gradients zero
        self.disc_optimizer.zero_grad()
        gen_loss = self._gen_step(condition, real)
        #self.tracker.track_loss("generator/loss", gen_loss)
        gen_loss.backward()
        self.gen_optimizer.step()

        loss = (disc_loss + gen_loss) / 2

        if display: # if the display value = true, which is every 100th iteration
            fake = self.gen(condition, conditioned_images_UP).detach()
            # Returns a new Tensor, detached from the current graph. The result will never require gradient.
            # This method also affects forward mode AD gradients and the result will never have forward mode AD gradients.
            display_progress(real[0], condition[0], fake[0])

        return loss, disc_loss, gen_loss

    def eval(self):
        for l in self.learner:
            l.eval()

    def train(self):
        for l in self.learner:
            l.train()