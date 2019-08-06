# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 22:23:49 2019

@author: memo
"""

from data_augment import *
from skimage.io import imread
import matplotlib.pyplot as plt

a = 'C:/Users/memo/Desktop/DeepLearningChangeDetection/Onera Satellite Change Detection dataset - Train Labels/valencia/cm/cm.png'

img = imread(a)

plt.imshow( paddingReflect(img, 200, 450, 200, 450, type =3) )