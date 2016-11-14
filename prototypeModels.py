from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import tensorflow as tf
import numpy as np


# # Define custom API for creating and adding layers to NN Model

def conv2d_layer(w_shape, prev_layer_out, layer_stride, w_name, if_relu):
    w_conv = tf.Variable(tf.random_normal(w_shape, stddev=0.35),
                      name=w_name)
    nextLayer = None
    if if_relu:
        nextLayer = tf.nn.relu(tf.nn.conv2d(prev_layer_out, w_conv, 
                        strides=layer_stride, padding='SAME',use_cudnn_on_gpu=False, name=w_name))
    else:
        nextLayer = tf.nn.conv2d(prev_layer_out, w_conv, 
                        strides=layer_stride, padding='SAME',use_cudnn_on_gpu=False,name=w_name)
    
    return nextLayer


def conv3d_layer(w_shape, prev_layer_out, layer_stride, w_name, if_relu):
    w_conv = tf.Variable(tf.random_normal(w_shape, stddev=0.35),
                      name=w_name)
    nextLayer = None
    if if_relu:
        nextLayer = tf.nn.relu(tf.nn.conv3d(prev_layer_out, w_conv, 
                        strides=layer_stride, padding='SAME',use_cudnn_on_gpu=False,name=w_name))
    else:
        nextLayer = tf.nn.conv3d(prev_layer_out, w_conv, 
                        strides=layer_stride, padding='SAME',use_cudnn_on_gpu=False,name=w_name)
    
    return nextLayer
    
def max_pool(prev_layer, window_size, str_size):
    next_layer = tf.nn.max_pool(prev_layer, ksize=window_size,
                        strides=str_size, padding='SAME')
    return next_layer

def dropout(prev_layer,  p_keep):
    next_layer = tf.nn.dropout(prev_layer, p_keep)
    return next_layer

def relu(prev_layer, window_size, str_size):
    next_layer = tf.nn.relu(prev_layer)
    return next_layer




