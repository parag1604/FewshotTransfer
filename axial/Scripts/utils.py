# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 13:12:05 2020

@author: Mehak


"""

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.applications.vgg19 import VGG19, preprocess_input

def affine_transform(inp):

    features = inp[0]
    alpha = inp[1]
    beta = inp[2]
    
    return alpha * (features) + beta

def binary_activation(x):

    cond = tf.less(x, tf.fill(tf.shape(x),0.5))
    out = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))

    return out

def dice_loss(y_true, y_pred, smooth = 1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1- ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f)+ smooth))



def dice_coef(y_true, y_pred, smooth = 1):
    
        y_true_thresh = K.greater_equal(y_true,0.5) #will return boolean values
        y_true_thresh = K.cast(y_true_thresh, dtype='float64')
        y_pred_thresh = K.greater_equal(y_pred,0.5) #will return boolean values
        y_pred_thresh = K.cast(y_pred_thresh, dtype='float64')
        y_true_f = K.flatten(y_true_thresh)
        y_pred_f = K.flatten(y_pred_thresh)
        intersection = K.sum(y_true_f * y_pred_f)
        
        return (2. * intersection + smooth ) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    

def evaluation_metric(pred, gt, threshold = 0.5):
    
    pred = np.int32(pred > threshold)
    diff = gt - pred
    false_positives = np.sum(diff == -1)
    false_negatives = np.sum(diff == 1)
    true_positives = np.sum(np.logical_and(pred, gt))
    true_negatives = np.sum(np.logical_and(np.logical_not(pred),np.logical_not(gt)))
    gt_positives = np.sum(gt)
    
    assert gt_positives == true_positives + false_negatives
    
    return false_positives, false_negatives, true_positives, true_negatives

def generate_fake_samples(g_model, samples, patch_shape):
    
    # generate fake instance
    X = g_model.predict(samples)
    # create 'fake' class labels (0)
    y = np.full((len(X), patch_shape, patch_shape, 1), tf.random.uniform(shape = [], minval = 0, maxval = 0.1))
    
    return X, y

def vgg_loss(feature_extractor):
    """
    Returns perceptual loss function with feature extractor as VGG19 trained on imagenet database
    """
    
    def perceptual_loss(y_true, y_pred):
        
        y_true_triple = K.concatenate((y_true, y_true, y_true), axis = -1)
        y_pred_triple = K.concatenate((y_pred, y_pred, y_pred), axis = -1)
        
        y_true_vgg = preprocess_input(y_true_triple)
        y_pred_vgg = preprocess_input(y_pred_triple)
        
        vgg_out_true = feature_extractor.model(y_true_vgg)
        vgg_out_pred = feature_extractor.model(y_pred_vgg)
        
        l1_loss = tf.keras.losses.MeanAbsoluteError()
        ltot= [l1_loss(vgg_out_pred[i], vgg_out_true[i]) for i in range(len(vgg_out_pred))]
        
        return K.sum(ltot)/len(vgg_out_pred)
    
    return perceptual_loss


class Cut_VGG19:

    #@misc{cardinale2018isr,
    #  title={ISR},
    #  author={Francesco Cardinale et al.},
    #  year={2018},
    #  howpublished={\url{https://github.com/idealo/image-super-resolution}},
    #}
    #Args:
    #    layers_to_extract: list of layers to be declared as output layers.
    #    patch_size: integer, defines the size of the input (patch_size x patch_size).
    #Attributes:
    #    loss_model: multi-output vgg architecture with <layers_to_extract> as output layers.

    def __init__(self, patch_size, layers_to_extract):
        self.patch_size = patch_size
        self.input_shape = (patch_size,) * 2 + (3,)
        self.layers_to_extract = layers_to_extract
        
        if len(self.layers_to_extract) > 0:
            self._cut_vgg()
        else:
            print('Invalid VGG instantiation: extracted layer must be > 0')
            raise ValueError('Invalid VGG instantiation: extracted layer must be > 0')
    
    def _cut_vgg(self):
        """
        Loads pre-trained VGG, declares as output the intermediate
        layers selected by self.layers_to_extract.
        """
        
        vgg = VGG19(weights='imagenet', include_top=False, input_shape=self.input_shape)
        vgg.trainable = False
        

        outputs = [vgg.layers[i].output for i in self.layers_to_extract]
        self.model = Model([vgg.input], outputs)
        self.model._name = 'feature_extractor'
        self.name = 'vgg19'  # used in weights naming
        
def calc_feature_mean_var(features):
    
    assert len(features.shape) == 4
    
    eps = tf.constant(1e-5)
    var = K.var(features, axis = [1,2]) + eps
    std = K.sqrt(var)
    m = K.mean(features, axis = [1,2])
    
    return m, std


