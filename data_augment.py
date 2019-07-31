# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 14:55:01 2019

@author: halid
"""

import numpy as np
from scipy.misc import imresize, imrotate
from skimage import exposure
from scipy.ndimage import gaussian_filter
import tensorflow as tf

def Color2Gray(img):
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    img[:,:,0], img[:,:,1], img[:,:,2] = gray, gray, gray
    
    return img

def SwapChannels(img,x=0,y=2):
    
    (img[:,:x], img[:,:,y]) = (img[:,:,y], img[:,:,x])
    
    return img

def flipv(img):
    
    return img[:,::-1,:]

def fliph(img):
    
    return img[::-1,:,:]

# heightxwidthxchannel
def resize(img, row, col):
    
    img = imresize(img, (row, col))
    
    return img

def normalize(img,x="norm8bit"):
    if x == "max":
        
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        rma = np.max(r)
        gma = np.max(g)
        bma = np.max(b)
        img = img/[rma, gma, bma]
        
        return img
    
    elif x == "minmax":
        
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        rm = np.min(r)
        gm = np.min(g)
        bm = np.min(b)
        rma = np.max(r)
        gma = np.max(g)
        bma = np.max(b)
        img = (img - [rm, gm, bm])/[rma-rm, bma-bm, gma-gm]
        
    elif x == "norm8bit":
        return img/255

def equalize(img, bins):
    
    exposure.equalize_hist(img, nbins = bins)
    
    return img 

def histo(img, bins):
    
    return exposure.histogram(img, nbins = bins)

def rotate(img, angle, interpolation='cubic'):
    
    imgnew = imrotate(img, angle, interp=interpolation)
    
    return imgnew

"""     'constant'
            Pads with a constant value.
        'edge'
            Pads with the edge values of array.
        'linear_ramp'
            Pads with the linear ramp between end_value and the
            array edge value.
        'maximum'
            Pads with the maximum value of all or part of the
            vector along each axis.
        'mean'
            Pads with the mean value of all or part of the
            vector along each axis.
        'median'
            Pads with the median value of all or part of the
            vector along each axis.
        'minimum'
            Pads with the minimum value of all or part of the
            vector along each axis.
        'reflect'
            Pads with the reflection of the vector mirrored on
            the first and last values of the vector along each
            axis.
        'symmetric'
            Pads with the reflection of the vector mirrored
            along the edge of the array.
        'wrap'
            Pads with the wrap of the vector along the axis.
            The first values are used to pad the end and the
            end values are used to pad the beginning.
        <function>
            Padding function, see Notes."""

def padding(img, width, mode):
    
    img = np.pad(img, width, mode)
    
    return img

def gausfilt(img, sig=1):
    imgnew = gaussian_filter(img, sigma=sig)
    
    return imgnew

def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)

def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2by2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
    
def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)

def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b