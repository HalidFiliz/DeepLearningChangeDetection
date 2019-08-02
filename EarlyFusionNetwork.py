import inspect
import os

import numpy as np
import tensorflow as tf
import time

class earlyFusionNetwork:
    def __init__(self):
         print("class created")
         
    def build(self, im1, nclass):
	
        start_time = time.time()
        print("build model started")

        self.conv_1_1 = self.convBatchnormRelu(im1, 16, 3, name= 'conv_1_1')
        self.conv_1_2 = self.convBatchnormRelu(self.conv_1_1, 16, 3, name= 'conv_1_2')
        self.maxPool_1= self.max_pool(self.conv_1_2, name='max_pool_1')
        
        self.conv_2_1 = self.convBatchnormRelu(self.maxPool_1, 32, 3, name= 'conv_2_1')
        self.conv_2_2 = self.convBatchnormRelu(self.conv_2_1, 32, 3, name= 'conv_2_2')
        self.maxPool_2= self.max_pool(self.conv_2_2, name='max_pool_2')
        
        self.conv_3_1 = self.convBatchnormRelu(self.maxPool_2, 64, 3, name= 'conv_3_1')
        self.conv_3_2 = self.convBatchnormRelu(self.conv_3_1, 64, 3, name= 'conv_3_2')
        self.conv_3_3 = self.convBatchnormRelu(self.conv_3_2, 64, 3, name= 'conv_3_3')
        self.maxPool_3= self.max_pool(self.conv_3_3, name='max_pool_3')
        
        self.conv_4_1 = self.convBatchnormRelu(self.maxPool_3, 128, 3, name= 'conv_4_1')
        self.conv_4_2 = self.convBatchnormRelu(self.conv_4_1, 128, 3, name= 'conv_4_2')
        self.conv_4_3 = self.convBatchnormRelu(self.conv_4_2, 128, 3, name= 'conv_4_3')
        self.maxPool_4= self.max_pool(self.conv_4_3, name='max_pool_4')
        
        self.convT_1 = self.convTranspose2d(self.maxPool_4, 128, 128, name='convT_1')
        self.concat_1 = self.concat(self.convT_1, self.conv_4_3, ax=-1)
        
        self.deconv_1_1 = self.convBatchnormRelu(self.concat_1, 128, 3, name= 'deconv_1_1')
        self.deconv_1_2 = self.convBatchnormRelu(self.deconv_1_1, 128, 3, name= 'deconv_1_2')
        self.deconv_1_3 = self.convBatchnormRelu(self.deconv_1_2, 64, 3, name= 'deconv_1_3')
        self.convT_2= self.convTranspose2d(self.deconv_1_3, 64, 64, name='convT_2')
        self.concat_2 = self.concat(self.convT_2, self.conv_3_3, ax=-1)
        
        self.deconv_2_1 = self.convBatchnormRelu(self.concat_2, 64, 3, name= 'deconv_2_1')
        self.deconv_2_2 = self.convBatchnormRelu(self.deconv_2_1, 64, 3, name= 'deconv_2_2')
        self.deconv_2_3 = self.convBatchnormRelu(self.deconv_2_2, 32, 3, name= 'deconv_2_3')
        self.convT_3= self.convTranspose2d(self.deconv_2_3, 32, 32, name='convT_3')
        self.concat_3 = self.concat(self.convT_3, self.conv_2_2, ax=-1)
        
        self.deconv_3_1 = self.convBatchnormRelu(self.concat_3, 32, 3, name= 'deconv_3_1')
        self.deconv_3_2 = self.convBatchnormRelu(self.deconv_3_1, 16, 3, name= 'deconv_3_2')
        self.convT_4= self.convTranspose2d(self.deconv_3_2, 16, 16, name='convT_4')
        self.concat_4 = self.concat(self.convT_4, self.conv_1_2, ax=-1)
        
        self.deconv_4_1 = self.convBatchnormRelu(self.concat_4, 16, 3, name= 'deconv_4_1')
        self.deconv_4_2 = self.convBatchnormRelu(self.deconv_4_1, 2, 3, name= 'deconv_4_2')
        
        
        
        print(("build model finished: %ds" % (time.time() - start_time)))

    def concat(self, layer1, layer2, ax=-1):
        return tf.concat([layer1, layer2], ax)  
    
    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    # def conv_layer(self, bottom, name):
        # with tf.variable_scope(name):
            # filt = self.get_conv_filter(name)

            # conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            # conv_biases = self.get_bias(name)
            # bias = tf.nn.bias_add(conv, conv_biases)

            # relu = tf.nn.relu(bias)
            # return relu

    # def fc_layer(self, bottom, name):
        # with tf.variable_scope(name):
            # shape = bottom.get_shape().as_list()
            # dim = 1
            # for d in shape[1:]:
                # dim *= d
            # x = tf.reshape(bottom, [-1, dim])

            # weights = self.get_fc_weight(name)
            # biases = self.get_bias(name)

            # # Fully connected layer. Note that the '+' operation automatically
            # # broadcasts the biases.
            # fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            # return fc

    # def get_conv_filter(self, name):
        # return tf.constant(self.data_dict[name][0], name="filter")

    # def get_bias(self, name):
        # return tf.constant(self.data_dict[name][1], name="biases")

    # def get_fc_weight(self, name):
        # return tf.constant(self.data_dict[name][0], name="weights")

	
    def conv2d(self, input_, output_dim, kernel=5, stride=2, stddev=0.02, name="conv2d", padding='SAME', isbias=True):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [kernel, kernel, input_.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.conv2d(input_, w, strides=[1, stride, stride, 1], padding=padding)

            if isbias == True:
                biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
                conv = tf.nn.bias_add(conv, biases)
			
        return conv
	
    def conv2dRelu(self, input_, output_dim, kernel=5, stride=2, stddev=0.02, name="conv2d", padding='SAME', isbias=True):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [kernel, kernel, input_.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.conv2d(input_, w, strides=[1, stride, stride, 1], padding=padding)

            if isbias == True:
                biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
                conv = tf.nn.bias_add(conv, biases)
		
        conv = self.relu(conv, name=name + "relu")
        return conv	
        
    def convBatchnormRelu(self, input_, output_dim, kernel=3, name='convBatchnormRelu'):
        conv = self.conv2d(input_, output_dim, kernel, stride=1 ,name=name+'conv2d')
        conv = self.batch_norm(conv, name=name+'batchNorm')
        conv = self.relu(conv, name=name+'Relu')
        return conv
    
    def convTranspose2d(self, x,  middle_channel, out_channel, name):
        x = self.conv2dRelu(x, output_dim=middle_channel, kernel=3, stride=1, name=name + '_conv2d_relu')
        x = self.deconv2d(x, [x.get_shape().as_list()[0], x.get_shape().as_list()[1]*2, x.get_shape().as_list()[2]*2, out_channel], kernel = 3, stride=2, name=name + '_deconv2d')
        x = self.relu(x)
        return x
	
    def deconv2d(self, input_, output_shape, kernel=5, stride=2, stddev=0.02, name="deconv2d"):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [kernel, kernel, output_shape[-1], input_.get_shape()[-1]], initializer=tf.random_normal_initializer(stddev=stddev))
    
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, stride, stride, 1])

            biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
            deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        return deconv

    def batch_norm(self, x, epsilon=1e-5, momentum = 0.9, name="batch_norm", training=True):
#        return tf.contrib.layers.batch_norm(x,
#						  decay=momentum, 
#						  updates_collections=None,
#						  epsilon=epsilon,
#						  scale=True,
#						  is_training=training,
#						  scope=name)
        
        return tf.layers.batch_normalization(x, training=training)
					  
    def prelu(self, x, name='prelu'):
        with tf.variable_scope(name):
            beta = tf.get_variable('beta', [x.get_shape()[-1]], tf.float32, 
									initializer=tf.constant_initializer(0.01))
		
        beta = tf.minimum(0.2, tf.maximum(beta, 0.01))
			
        return tf.maximum(x, beta*x)
		
		
    def instance_norm(self,x, name='const_norm'):
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, 1e-9)))

    def channel_norm(self,x, name='channel_norm'):
        mean, var = tf.nn.moments(x, [1, 2, 3], keep_dims=True)
        return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, 1e-9)))

    def dropout(self,x, keep_prob=0.5, training=True):
        #prob = tf.cond(training, keep_prob, 1.0)
        return tf.nn.dropout(x, keep_prob=keep_prob)

    def lrelu(self,x, leak=0.2, name='lrelu'):
        return tf.maximum(x, leak*x)

    def relu(self,x, name='relu'):
        return tf.nn.relu(x)