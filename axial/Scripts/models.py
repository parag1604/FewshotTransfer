# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 07:34:12 2020

@author: Mehak

Models

"""
import tensorflow as tf

from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import numpy as np
from utils import *


def downsample_layer(layer_input, n_filters, filter_size = 3, stride = 1):
    init = RandomNormal(stddev=0.02)
    return Conv2D(n_filters, (filter_size,filter_size), strides=(stride,stride), padding='same', kernel_initializer=init)(layer_input)
   
def upsample_layer(layer_input, n_filters, filter_size = 3, stride = 1):
    init = RandomNormal(stddev=0.02)
    return Conv2DTranspose(n_filters, (filter_size,filter_size), strides=(stride,stride), padding='same', kernel_initializer=init)(layer_input)  

def param_gen_bottleneck(features, bottleneck_depth, n_filters):
    init = RandomNormal(stddev = 0.2)
    
    b1 = Conv2D(bottleneck_depth, (1,1), kernel_initializer = init)(features)
    a1 = Activation('relu')(b1)
    b2 = Conv2D(n_filters, (1,1), kernel_initializer = init)(a1)
    
    return b2


def encoder_block_withBFT(target_, guide_, n_filters, filter_size = 3, stride = 1, bottleneck_depth = 100, BFT=True):

    target_feat = downsample_layer(target_, n_filters, filter_size, stride)
    guide_feat = downsample_layer(guide_, n_filters, filter_size, stride)
    
    if BFT:
        target_alpha = param_gen_bottleneck(target_feat, bottleneck_depth, n_filters)
        target_beta = param_gen_bottleneck(target_feat, bottleneck_depth, n_filters)
        
        guide_alpha = param_gen_bottleneck(guide_feat, bottleneck_depth, n_filters)
        guide_beta = param_gen_bottleneck(guide_feat, bottleneck_depth, n_filters)
        
        target_feat = LayerNormalization(scale = False, center = False)(target_feat)
        guide_feat = LayerNormalization(scale = False, center = False)(guide_feat)
        
        Affine1 = Lambda(affine_transform)
        target_feat = Affine1([target_feat, guide_alpha, guide_beta])
        Affine2 = Lambda(affine_transform)
        guide_feat = Affine2([guide_feat, target_alpha, target_beta])
        
    target_out = LeakyReLU(alpha=0.2)(target_feat)
    guide_out = LeakyReLU(alpha=0.2)(guide_feat)

    return target_out, guide_out

def decoder_block_layernorm(layer_in, skip_in, n_filters, filter_size = 3, stride = 1, dropout=True):
    
    g = upsample_layer(layer_in, n_filters, filter_size, stride)     
    g = LayerNormalization(axis = -1)(g)
    
    # conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)

    g = Concatenate()([g, skip_in])
    g = Activation('relu')(g)
    return g

def encoder_block_batchnorm(layer_in, n_filters, filter_size = 3, stride = 1, norm=True):

    g = downsample_layer(layer_in, n_filters, filter_size, stride)
    if norm:
        g = BatchNormalization(axis = -1)(g)
    g = LeakyReLU(alpha=0.2)(g)
    return g

def decoder_block_batchnorm(layer_in, skip_in, n_filters, filter_size = 3, stride = 1, dropout=True):
	
    g = upsample_layer(layer_in, n_filters, filter_size, stride)
    g = BatchNormalization(axis = -1)(g)

    # conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
    g = Concatenate()([g, skip_in])
    g = Activation('relu')(g)
    return g

def unet(image_shape, activation_fn = 'tanh'):

    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=image_shape)
    
    # encoder model: C32-C64-C128-C256-C512
    e1 = encoder_block_batchnorm(in_image, 32, norm=False)
    e1 = encoder_block_batchnorm(e1, 32, stride = 2)
    
    e2 = encoder_block_batchnorm(e1, 64)
    e2 = encoder_block_batchnorm(e2, 64, stride = 2)
    
    e3 = encoder_block_batchnorm(e2, 128)
    e3 = encoder_block_batchnorm(e3, 128, stride = 2)
    
    e4 = encoder_block_batchnorm(e3, 256)
    e4 = encoder_block_batchnorm(e3, 256, stride = 2)
    
    # bottleneck, no batch norm and relu
    b = Conv2D(512, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(e4)
    b = Activation('relu')(b)
    
    # decoder model: CD512-CD1024-C512-C256-C128-C64
    d1 = decoder_block_batchnorm(b, e4, 256, stride = 2)
    d1 = Conv2D(256, (3,3), activation = 'relu', padding = 'same', kernel_initializer = init)(d1)
    d1 = BatchNormalization(axis = -1)(d1)
    d1 = Dropout(0.5)(d1, training=True)
    
    d2 = decoder_block_batchnorm(d1, e3, 128, stride = 2)
    d2 = Conv2D(128, (3,3), activation = 'relu', padding = 'same', kernel_initializer = init)(d2)
    d2 = BatchNormalization(axis = -1)(d2)
    d2 = Dropout(0.5)(d2, training=True)
    
    d3 = decoder_block_batchnorm(d2, e2, 64, stride = 2, dropout = False)
    d3 = Conv2D(64, (3,3), activation = 'relu', padding = 'same', kernel_initializer = init)(d3)
    d3 = BatchNormalization(axis = -1)(d3)
    
    d4 = decoder_block_batchnorm(d3, e1, 32, stride = 2, dropout=False)
    d4 = Conv2D(32, (3,3), activation = 'relu', padding = 'same', kernel_initializer = init)(d4)
    d4 = BatchNormalization(axis = -1)(d4)
    
    # output
    g = Conv2DTranspose(1, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(d4)
    out_image = Activation(activation_fn)(g)
    	
    model = Model(in_image, out_image, name = 'Cpp_unet')
    
    return model


def unet_withBFT(image_shape=(256,256,1), activation_fn = 'tanh'):

    init = RandomNormal(stddev=0.02)
    in_target = Input(shape=image_shape, name='target_image')
    in_guide = Input(shape=image_shape, name='guide_image')
    
    # encoder model: C32-C64-C128-C256-C512
    t1, g1 = encoder_block_withBFT(in_target, in_guide, 32, bottleneck_depth = 100, BFT=False)
    t1, g1 = encoder_block_withBFT(t1, g1, 32, stride = 2, bottleneck_depth = 100)
    
    t2, g2 = encoder_block_withBFT(t1, g1, 64, bottleneck_depth = 100)
    t2, g2 = encoder_block_withBFT(t2, g2, 64, stride = 2, bottleneck_depth = 100)
    
    t3, g3 = encoder_block_withBFT(t2, g2, 128, bottleneck_depth = 100)
    t3, g3 = encoder_block_withBFT(t3, g3, 128, stride = 2, bottleneck_depth = 100)
    
    t4, g4 = encoder_block_withBFT(t3, g3, 256, bottleneck_depth = 100)
    t4, g4 = encoder_block_withBFT(t4, g4, 256, stride = 2, bottleneck_depth = 100)

    # bottleneck, no batch norm and relu
    b = Conv2D(512, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(t4)
    b = Activation('relu')(b)

    # decoder model: CD1024-C512-C256-C128
    d1 = decoder_block_layernorm(b, t4, 256, stride = 2)
    d1 = Conv2D(256, (3,3), activation = 'relu', padding = 'same', kernel_initializer = init)(d1)
    d1 = LayerNormalization(axis = -1)(d1)
    d1 = Dropout(0.5)(d1, training=True)
    
    d2 = decoder_block_layernorm(d1, t3, 128, stride = 2)
    d2 = Conv2D(128, (3,3), activation = 'relu', padding = 'same', kernel_initializer = init)(d2)
    d2 = LayerNormalization(axis = -1)(d2)
    d2 = Dropout(0.5)(d2, training=True)
    
    d3 = decoder_block_layernorm(d2, t2, 64, stride = 2, dropout = False)
    d3 = Conv2D(64, (3,3), activation = 'relu', padding = 'same', kernel_initializer = init)(d3)
    d3 = LayerNormalization(axis = -1)(d3)
    
    d4 = decoder_block_layernorm(d3, t1, 32, stride = 2, dropout=False)
    d4 = Conv2D(32, (3,3), activation = 'relu', padding = 'same', kernel_initializer = init)(d4)
    d4 = LayerNormalization(axis = -1)(d4)
    
    # output
    g = Conv2DTranspose(1, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(d4)
    out_image = Activation(activation_fn)(g)
    
    model = Model([in_target, in_guide], out_image, name = 'Generator_bft')
            
    return model

def gpp_generator(image_shape=(256,256,1), activation_fn = 'tanh'):

    init = RandomNormal(stddev=0.02)
    in_target = Input(shape=image_shape, name='target_image')
    in_guide = Input(shape=image_shape, name='guide_image')

    # encoder model: C64-C128-C256-C512-C512-C512-C512-C512
    t1, g1 = encoder_block_withBFT(in_target, in_guide, 64, 4, 2, 100, BFT=False)
    t2, g2 = encoder_block_withBFT(t1, g1, 128, 4,2, 100)
    t3, g3 = encoder_block_withBFT(t2, g2, 256, 4,2, 100)
    t4, g4 = encoder_block_withBFT(t3, g3, 512, 4,2, 100)
    t5, g5 = encoder_block_withBFT(t4, g4, 512, 4,2, 100)
    t6, g6 = encoder_block_withBFT(t5, g5, 512, 4,2, 100)
    t7, g7 = encoder_block_withBFT(t6, g6, 512, 4,2, 100)

    # bottleneck, no batch norm and relu
    b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(t7)
    b = Activation('relu')(b)

    # decoder model: CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
    d1 = decoder_block_layernorm(b, t7, 512, 4, 2)
    d2 = decoder_block_layernorm(d1, t6, 512, 4, 2)
    d3 = decoder_block_layernorm(d2, t5, 512, 4, 2 )
    d4 = decoder_block_layernorm(d3, t4, 512, 4, 2, dropout=False)
    d5 = decoder_block_layernorm(d4, t3, 256, 4, 2, dropout=False)
    d6 = decoder_block_layernorm(d5, t2, 128, 4, 2, dropout=False)
    d7 = decoder_block_layernorm(d6, t1, 64, 4, 2, dropout=False)

    # output
    g = Conv2DTranspose(1, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
    out_image = Activation(activation_fn)(g)
    
    model = Model([in_target, in_guide], out_image, name = 'Generator_bft')
            
    return model

def unet_p2p(image_shape, activation_fn = 'tanh'):

    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=image_shape)
    
    # encoder model: C64-C128-C256-C512-C512-C512-C512-C512
    e1 = encoder_block_batchnorm(in_image, 64, norm=False)
    e2 = encoder_block_batchnorm(e1, 128)
    e3 = encoder_block_batchnorm(e2, 256)
    e4 = encoder_block_batchnorm(e3, 512)
    e5 = encoder_block_batchnorm(e4, 512)
    e6 = encoder_block_batchnorm(e5, 512)
    e7 = encoder_block_batchnorm(e6, 512)
    
    # bottleneck, no batch norm and relu
    b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
    b = Activation('relu')(b)
    
    # decoder model: CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
    d1 = decoder_block_batchnorm(b, e7, 512)
    d2 = decoder_block_batchnorm(d1, e6, 512)
    d3 = decoder_block_batchnorm(d2, e5, 512)
    d4 = decoder_block_batchnorm(d3, e4, 512, dropout=False)
    d5 = decoder_block_batchnorm(d4, e3, 256, dropout=False)
    d6 = decoder_block_batchnorm(d5, e2, 128, dropout=False)
    d7 = decoder_block_batchnorm(d6, e1, 64, dropout=False)
    
    # output
    g = Conv2DTranspose(1, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
    out_image = Activation(activation_fn)(g)
    	
    model = Model(in_image, out_image, name = 'Cpp_unet')
    
    return model

def discriminator(image_shape):
  
    #Weight Initialisation
    init = RandomNormal(stddev= 0.02)

    #Inputs
    in_img = Input(shape=image_shape, name='input_image')
    in_seg = Input(shape=image_shape, name='target_image')
    
    #binarise the segmentation predicted
    binarize = Lambda(binary_activation)
    in_seg_thresh = binarize(in_seg)

    x = Concatenate()([in_img, in_seg_thresh]) # (bs, 256, 256, channels*2)

    d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(x)
    d = LeakyReLU(alpha=0.2)(d)

    # C128
    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    # C256
    d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    zero_pad1 = ZeroPadding2D()(d) # (bs, 34, 34, 256)

    # C512
    d = Conv2D(512, (4,4), strides=1, kernel_initializer=init)(zero_pad1)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    zero_pad2 = ZeroPadding2D()(d) # (bs, 33, 33, 512)

    last = Conv2D(1, 4, strides=1, kernel_initializer=init)(zero_pad2) # (bs, 30, 30, 1)
    patch_out = LeakyReLU(alpha = 0.2)(last)

    
    model = Model([in_img, in_seg], patch_out, name = 'Discriminator')
  
    return model

def guided_gan(g_model, d_model, image_shape, image_guide, mask_guide):
	
    # make weights in the discriminator not trainable
    d_model.trainable = False
    
    in_target = Input(shape=image_shape, name='input_image')
    in_guide = Input(shape=image_shape, name='input_mask')
    
    if(not (image_guide or mask_guide)):
        print("Taking mask as the guide image")
        mask_guide = True
    
    if(mask_guide):
        gen_in = [in_target, in_guide]
    elif(image_guide):
        gen_in = [in_guide, in_target]
        
    gen_out = g_model(gen_in)

    dis_in = [in_target, gen_out]
    dis_out = d_model(dis_in)
    
    #GAN model -  condition at the input, generated image and PATCHGAN classification 
    model = Model([in_target, in_guide], [dis_out, gen_out], name='GAN')
    
    return model