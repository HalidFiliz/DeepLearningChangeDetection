# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 20:22:14 2019

@author: halid
utils
"""
import numpy as np
import tensorflow as tf

# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# l2_decay : tf_contrib.layers.l2_regularizer(0.0001)

weight_init = tf_contrib.layers.variance_scaling_initializer()
weight_regularizer = tf_contrib.layers.l2_regularizer(0.0001)

def concat(  layer1, layer2, ax=-1):
	return tf.concat([layer1, layer2], ax)  

def avg_pool(  bottom, name):
	return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def max_pool(  bottom, name):
	return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def conv2d(  input_, output_dim, kernel=5, stride=2, stddev=0.02, name="conv2d", padding='SAME', isbias=True):
	with tf.variable_scope(name):
		w = tf.get_variable('w', [kernel, kernel, input_.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=stddev))
		conv = tf.nn.conv2d(input_, w, strides=[1, stride, stride, 1], padding=padding)

		if isbias == True:
			biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
			conv = tf.nn.bias_add(conv, biases)
		
	return conv

def batch_norm(  x, epsilon=1e-5, momentum = 0.9, name="batch_norm", training=True):
	return tf.layers.batch_normalization(x, training=training)

def deconv2d(  input_, output_shape, kernel=5, stride=2, stddev=0.02, name="deconv2d"):
	with tf.variable_scope(name):
		w = tf.get_variable('w', [kernel, kernel, output_shape[-1], input_.get_shape()[-1]], initializer=tf.random_normal_initializer(stddev=stddev))

		deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, stride, stride, 1])

		biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
		deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

	return deconv
	
def prelu(  x, name='prelu'):
	with tf.variable_scope(name):
		beta = tf.get_variable('beta', [x.get_shape()[-1]], tf.float32, 
								initializer=tf.constant_initializer(0.01))
	
	beta = tf.minimum(0.2, tf.maximum(beta, 0.01))
		
	return tf.maximum(x, beta*x)
	
	
def instance_norm( x, name='const_norm'):
	mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
	return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, 1e-9)))

def channel_norm( x, name='channel_norm'):
	mean, var = tf.nn.moments(x, [1, 2, 3], keep_dims=True)
	return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, 1e-9)))

def dropout( x, keep_prob=0.5, training=True):
	#prob = tf.cond(training, keep_prob, 1.0)
	return tf.nn.dropout(x, keep_prob=keep_prob)

def lrelu( x, leak=0.2, name='lrelu'):
	return tf.maximum(x, leak*x)

def relu( x, name='relu'):
	return tf.nn.relu(x)

def swish(x, name='swish'):
	return x*tf.nn.sigmoid(x)

def subpixel(X, r, n_out_channel):
    if n_out_channel >= 1:
        assert int(X.get_shape()[-1]) == (r ** 2) * n_out_channel, 'Invalid Params'
        bsize, a, b, c = X.get_shape().as_list()
        bsize = tf.shape(X)[0] # Handling Dimension(None) type for undefined batch dim
        Xs=tf.split(X,r,3) #b*h*w*r*r
        Xr=tf.concat(Xs,2) #b*h*(r*w)*r
        X=tf.reshape(Xr,(bsize,r*a,r*b,n_out_channel)) # b*(r*h)*(r*w)*c
    else:
        print('Invalid Dim.')
    return X
	
################################################### FC
def fully_conneted(x, units, use_bias=True ):
        x = flatten(x)
        
        x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, 
                            kernel_regularizer=weight_regularizer, use_bias=use_bias
                           )

        return x
		
################################################### custom designed layers
def conv2dRelu(  input_, output_dim, kernel=5, stride=2, stddev=0.02, name="conv2d", padding='SAME', isbias=True):
	with tf.variable_scope(name):
		w = tf.get_variable('w', [kernel, kernel, input_.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=stddev))
		conv = tf.nn.conv2d(input_, w, strides=[1, stride, stride, 1], padding=padding)

		if isbias == True:
			biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
			conv = tf.nn.bias_add(conv, biases)
	
	conv =  relu(conv, name=name + "relu")
	return conv	
	
def convBatchnormRelu(  input_, output_dim, kernel=3, name='convBatchnormRelu'):
	conv =  conv2d(input_, output_dim, kernel, stride=1 ,name=name+'conv2d')
	conv =  batch_norm(conv, name=name+'batchNorm')
	conv =  relu(conv, name=name+'Relu')
	return conv

def convTranspose2d(  x,  middle_channel, out_channel, name):
	x =  conv2dRelu(x, output_dim=middle_channel, kernel=3, stride=1, name=name + '_conv2d_relu')
	x =  deconv2d(x, [x.get_shape().as_list()[0], x.get_shape().as_list()[1]*2, x.get_shape().as_list()[2]*2, out_channel], kernel = 3, stride=2, name=name + '_deconv2d')
	x =  relu(x)
	return x