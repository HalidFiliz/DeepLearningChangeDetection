# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 20:19:57 2019

@author: halid
train
"""

import tensorflow as tf
import numpy as np
from scipy.misc import imsave as imsave

import EarlyFusionNetwork
from losses import *

tf.reset_default_graph()

checkpoint_dir = './checkpoints'
log_dir = './logs'
model_name= 'EarlyFusion'
summary_counter = 1

learning_rate  = 0.008
epoch = 200
batch_size = 64
patience = 5

width   = 112
height  = 112
channel = bat.giveChannelCount()
nclass  = 2

iteration = bat.giveSize() // batch_size

def model_dir():
    return "{}_{}".format(model_name, batch_size)

def save(checkpoint_dir, step):
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir())

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(sess, os.path.join(checkpoint_dir, model_name+'.model'), global_step=step)

def load(checkpoint_dir):
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir())

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        counter = int(ckpt_name.split('-')[-1])
        print(" [*] Success to read {}".format(ckpt_name))
        return True, counter
    else:
        print(" [*] Failed to find a checkpoint")
        return False, 0
    
#%%
xi = tf.placeholder("float", [batch_size, width, height, channel])
xo = tf.placeholder("float", [batch_size, width, height, nclass])
lr = tf.placeholder("float", name="learning_rate")
keep_prob = tf.placeholder("float", name='keep_prob')

efn = EarlyFusionNetwork.earlyFusionNetwork()
efn.build(xi, nclass, keep_prob)

xop  = efn.deconv_4_3
xops = tf.nn.softmax(xop, name='y_pred')

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=xo, logits=xop))

optimizerc = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
#optimizerc = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.995).minimize(loss)

train_loss =[]
valid_loss =[]
valid_good_loss =[]

saver = tf.train.Saver()

with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
                
        writer = tf.summary.FileWriter(log_dir + '/' + model_dir(), sess.graph)
        
        best_loss = 1.0
        notImproved = 0
        could_load, checkpoint_counter = load(checkpoint_dir)   
        if could_load:
            start_epoch = checkpoint_counter
            start_batch_id = 0
            summary_counter = checkpoint_counter

            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            summary_counter  = 1
            print(" [!] Load failed...")
            
        for epoch in range(start_epoch, epoch):
            
            if notImproved == patience:
                learning_rate = learning_rate*0.5
                notImproved = 0
                print("learnin rate decreased to " + str(learning_rate))
            
            for batsi in range(start_batch_id, int((bat.giveSize())/batch_size)):
                
                trdata, trlabel = bat.giveBatch(batch_size)
                _, cl  = sess.run([optimizerc, loss], feed_dict={xi: trdata, xo: trlabel, lr:learning_rate, keep_prob:0.8})            

            train_loss.append(cl)
            
            batch_xi, batch_xo = batv.giveBatch(batch_size)       
            pred, v_loss = sess.run([xops, loss], feed_dict={xi: batch_xi, xo: batch_xo, keep_prob:1.0})
            
            valid_loss.append(v_loss)
            
            for idx, im in enumerate(pred):
                imsave('test/img' + str(idx) + '_' + str(0) + '_pred.png', im[:,:,0]*255.)
                    
            for idx, im in enumerate(batch_xo):
                imsave('test/img' + str(idx) + '_' + str(0) + '_label.png', im[:,:,0]*255.)

            for idx, im in enumerate(batch_xi):
                imsave('test/img' + str(idx) + '_origRGB_1.png', color2Gray(im[...,0:3]))  
                imsave('test/img' + str(idx) + '_origRGB_2.png', color2Gray(im[...,3:6]))
            
            start_batch_id = 0
            summary_counter += 1
            if cl < best_loss:
                print("best loss is " + str(v_loss))
                best_loss = cl
                valid_good_loss.append(v_loss)
                
                save(checkpoint_dir, summary_counter)
                
                notImproved = 0
            else:
                print("loss did not improved")
                notImproved = notImproved + 1
                
            print("Epoch " + str(epoch) + ", Minibatch Los= " + \
                     "{:.6f}".format(cl) + " , Val Los= {:.6f}".format(v_loss) ) 
            
        save(checkpoint_dir, summary_counter)
            
plt.plot(train_loss)        
plt.plot(valid_loss)
plt.plot(valid_good_loss)

#%%
with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        could_load, checkpoint_counter = load(checkpoint_dir)
        w = width
        h = height
        hStep = 50
        wStep = 50
        for i in range(len(test_images)):
            target_img = test_images[i]
            output_image = np.zeros(( target_img.shape[0], target_img.shape[1]) , target_img.dtype)
            for hh in range(0, target_img.shape[0], hStep):
                hStart = 0
                hStop  = 0
                if (hh + h) > target_img.shape[0]: 
                    hStart = target_img.shape[0] - h
                    hStop = target_img.shape[0]
                else:
                    hStart = hh
                    hStop = hh + h
 
                for ww in range(0, target_img.shape[1], wStep):
                    wStart = 0
                    wStop  = 0
                    if (ww + w) > target_img.shape[1]: 
                        wStart = target_img.shape[1] - w
                        wStop = target_img.shape[1]
                    else:
                        wStart = ww
                        wStop = ww + w
                    
                    target_batch = np.zeros(( batch_size, height, width, channel) , target_img.dtype)
                    for idx in range(len(target_batch)):
                        target_batch[idx,...] = target_img[hStart:hStop, wStart:wStop, :]
                        
                    pred_test, = sess.run([xops], feed_dict={xi: target_batch, keep_prob:1.0})
                    output_image[hStart:hStop, wStart:wStop] = pred_test[0,...,0]
                    
            imsave('output/img' + str(i) + '_result.tif', output_image)