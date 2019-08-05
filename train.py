# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 20:19:57 2019

@author: halid
train
"""

import tensorflow as tf
from scipy.misc import imsave as imsave

import EarlyFusionNetwork
from losses import *

tf.reset_default_graph()

learning_rate  = 0.0001
epoch = 10
batch_size = 12
best_val = 100

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

#loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=xop, labels=xo))

loss, acc = classification_loss(logit=xop, label=xo)

optimizerc = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
train_loss =[]
valid_loss =[]
valid_acc =[]
with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        
        for epoch in range(epoch):
            for batsi in range(int((bat.giveSize())/batch_size)):
                trdata, trlabel = bat.giveBatch(batch_size)
                _, cl, ac  = sess.run([optimizerc, loss, acc], feed_dict={xi: trdata, xo: trlabel})            

            train_loss.append(cl)
            
            batch_xi, batch_xo = batv.giveBatch(batch_size)       
            pred, v_loss, v_acc = sess.run([xops, loss, acc], feed_dict={xi: batch_xi, xo: batch_xo})
            
            valid_loss.append(v_loss)
            valid_acc.append(v_acc)
            
            print("Epoch " + str(epoch) + ", Minibatch Los= " + \
                      "{:.6f}".format(cl) + " , Val Los= {:.6f}".format(v_loss) + " , Val Acc= {:.6f}".format(v_acc) )   
            
            for idx, im in enumerate(pred):
                imsave('test/img' + str(idx) + '_' + str(0) + '_pred.png', im[:,:,0]*255.)
                    
            for idx, im in enumerate(batch_xo):
                imsave('test/img' + str(idx) + '_' + str(0) + '_label.png', im[:,:,0]*255.)

            for idx, im in enumerate(batch_xi):
                imsave('test/img' + str(idx) + '_origRGB_1.png', color2Gray(im[...,0:3]))  
                imsave('test/img' + str(idx) + '_origRGB_2.png', color2Gray(im[...,3:6]))
            
            if v_loss <= best_val:
                best_val = v_loss
                saver.save(sess, 'Model_Backup/model.ckpt')
                print("Checkpoint created!")
        print('Loading pre-trained weights for the model...')
        saver = tf.train.Saver()
        saver.restore(sess, 'Model_Backup/model.ckpt')
        sess.run(tf.global_variables())
        print('\nRESTORATION COMPLETE\n')
            
plt.plot(train_loss)        
plt.plot(valid_loss)