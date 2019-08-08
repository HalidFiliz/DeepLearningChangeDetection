# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 14:55:01 2019

@author: halid
"""

import numpy as np
from scipy.misc import imresize, imrotate
from skimage import exposure
from scipy.ndimage import gaussian_filter

def color2Gray(img):
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    #img[:,:,0], img[:,:,1], img[:,:,2] = gray, gray, gray
    img[:,:,0]= gray
    return img[:,:,0]

def swapChannels(img,x=0,y=2):
    
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

def equalizeHisto(img, bins=256):
    exposure.equalize_hist(img, nbins = bins)
    return img 

#clahe
def equalizeAdaptHisto_multi(img):
    c = img.shape[-1]
    retMat = np.zeros( img.shape, img.dtype )
    for i in range(c):
        retMat[:,:,i] = equalizeAdaptHisto(img[:,:,i])
    
    return retMat

def equalizeAdaptHisto(img):
    return exposure.equalize_adapthist(img, clip_limit=0.03)

def histo(img, bins):
    return exposure.histogram(img, nbins = bins)

def rotate_multi(img, angle, interpolation='cubic'):
    c = img.shape[-1]
    retMat = np.zeros( img.shape, img.dtype )
    
    for i in range(c):
        retMat[:,:,i] = rotate(img[:,:,i], angle, interpolation)
    
    return retMat

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

def rescale_intensity_multi(img):
    c = img.shape[-1]
    retMat = np.zeros( img.shape, img.dtype ) 
    for i in range(c):
        retMat[:,:,i] = exposure.rescale_intensity(img[:,:,i])
    
    return retMat

def paddingReflect(img, hStart, hStop, wStart, wStop, type = 1): # type 0-1-2-3
    output = np.zeros(img.shape, img.dtype)
    if type == 0:
        output[0:(hStop-hStart), 0:(wStop-wStart), :] = img[hStart:hStop, wStart:wStop, :] 
        
        idx = (wStop-wStart)-1
        for ww in range((wStop-wStart), img.shape[1], 1):
            output[:,ww,:] = output[:,idx,:]
            idx -= 1
            
        idx = (hStop-hStart)-1
        for hh in range((hStop-hStart), img.shape[0], 1):
            output[hh,:,:] = output[idx,:,:]
            idx -= 1
    
    if type == 1:
        img = img[hStart:hStop, wStart:wStop, :] 
        output[0:img.shape[0] , (output.shape[1]-img.shape[1]):output.shape[1], :] = img
        
        idx = (output.shape[1]-img.shape[1])+1
        for ww in range( (output.shape[1]-img.shape[1]), 0, -1):
            output[:,ww,:] = output[:,idx,:]
            idx += 1
        
        idx = img.shape[0]-1
        for hh in range( img.shape[0], output.shape[0], 1):
            output[hh,:,:] = output[idx,:,:]
            idx -= 1
    
    if type == 2:
        img = img[hStart:hStop, wStart:wStop, :] 
        output[(output.shape[0]-img.shape[0]):output.shape[0] ,0:img.shape[1] , :] = img
        
        idx = (output.shape[0]-img.shape[0])+1
        for hh in range( (output.shape[0]-img.shape[0]), 0, -1):
            output[hh,:,:] = output[idx,:,:]
            idx += 1
      
        idx = img.shape[1]-1
        for ww in range( img.shape[1], output.shape[1], 1):
            output[:,ww,:] = output[:,idx,:]
            idx -= 1
            
    if type == 3:
        img = img[hStart:hStop, wStart:wStop, :] 
        output[(output.shape[0]-img.shape[0]):output.shape[0] , (output.shape[1]-img.shape[1]):output.shape[1], :] = img
            
        idx = (output.shape[1]-img.shape[1])+1
        for ww in range( (output.shape[1]-img.shape[1]), 0, -1):
            output[:,ww,:] = output[:,idx,:]
            idx += 1
        
        idx = (output.shape[0]-img.shape[0])+1
        for hh in range( (output.shape[0]-img.shape[0]), 0, -1):
            output[hh,:,:] = output[idx,:,:]
            idx += 1
            
    return output

def gausfilt(img, sig=1):
    imgnew = gaussian_filter(img, sigma=sig)
    
    return imgnew