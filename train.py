# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 20:19:57 2019

@author: halid
train
"""

import time
import tensorflow as tf
from scipy.misc import imsave as imsave

import EarlyFusionNetwork


tf.reset_default_graph()

learning_rate  = 0.0001
epoch = 20
batch_size = 18

width   = 224
height  = 224

channel = 12
nclass  = 2

xi = tf.placeholder("float", [batch_size, width, height, channel])
xo = tf.placeholder("float", [batch_size, width, height, nclass])
btrain = tf.placeholder("bool", None)

efn = EarlyFusionNetwork.earlyFusionNetwork()
efn.build(xi, nclass)

xop  = efn.deconv_4_2
xops = tf.nn.softmax(xop, name='y_pred')

loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=xop, labels=xo))

optimizerc = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
train_loss =[]
valid_loss =[]
valid_acc =[]

with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        

        
        for epoch in range(epoch):
            
            start_time = time.time()
            print("epoch started")
            
            for batsi in range(int((bat.giveSize())/batch_size)):
                trdata, trlabel = bat.giveBatch(batch_size)
                _, cl  = sess.run([optimizerc, loss], feed_dict={xi: trdata, xo: trlabel})
           
            print("Epoch " + str(epoch) + ", Minibatch Loss= " + \
                      "{:.6f}".format(cl))   
            
            train_loss.append(cl)
            print(("epoch finished: %ds" % (time.time() - start_time)))
            
            batch_xi, batch_xo = batv.giveBatch(batch_size)       
            pred = sess.run(xops, feed_dict={xi: batch_xi})
            
            for idx, im in enumerate(pred):
                for c in range(nclass):
                    imsave('test/img' + str(idx) + '_' + str(c) + '_pred.png', im[:,:,c]*255.)
                    
            for idx, im in enumerate(batch_xo):
                for c in range(nclass):
                    imsave('test/img' + str(idx) + '_' + str(c) + '_label.png', im[:,:,c]*255.)

            for idx, im in enumerate(batch_xi):
                imsave('test/img' + str(idx) + '_origRGB_1.png', color2Gray(im[...,0:3]))  
                imsave('test/img' + str(idx) + '_origRGB_2.png', color2Gray(im[...,3:6]))
            
            cv  = sess.run(loss, feed_dict={xi: batch_xi, xo: batch_xo})
            print("Epoch " + str(epoch) + ", Valid Loss= " + \
                      "{:.6f}".format(cv))
            valid_loss.append(cv)
            
            prediction = tf.equal(tf.argmax(batch_xo, -1), tf.argmax(pred, -1))
            accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
            acc  = sess.run(accuracy, feed_dict={xi: batch_xi, xo: batch_xo})
            print("Acc: "+"{:.6f}".format(acc))
            
plt.plot(train_loss)
plt.plot(valid_loss)        
        