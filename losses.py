# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 20:10:12 2019

@author: halid
loss functions
"""

import tensorflow as tf

def classification_loss(logit, label) :
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit))
	prediction = tf.equal(tf.argmax(logit, -1), tf.argmax(label, -1))
	accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

	return loss, accuracy
	

# https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/

def dice_loss(y_true, y_pred):
	denominator = tf.reduce_sum(y_true + tf.square(y_pred))
	numerator = 2 * tf.reduce_sum(y_true * y_pred) # some implementations don't square y_pred
	return numerator / (denominator + tf.keras.backend.epsilon())
    


#return tf.nn.tf.reduce_mean( numerator / (denominator + tf.keras.backend.epsilon()) )
    


def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)) - tf.log(dice_loss(y_true, y_pred))

def tversky_loss(beta):
	def loss(y_true, y_pred):
		numerator = tf.reduce_sum(y_true * y_pred)
		denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)

		return numerator / (tf.reduce_sum(denominator) + tf.keras.backend.epsilon())

	return loss
	
def focal_loss(alpha=0.25, gamma=2):
	def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
		weight_a = alpha * (1 - y_pred) ** gamma * targets
		weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)
    
		return (tf.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b 

	def loss(y_true, y_pred):
		y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
		logits = tf.log(y_pred / (1 - y_pred))

		loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)

		return tf.reduce_mean(loss)

	return loss