from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import tensorflow as tf
import numpy as np
from input_brain import *


# Define custom API for creating and adding layers to NN Model
# Wrapper around Tensorflow API, for ease of use and readibility

class ConvolutionalNN(object):

    def __init__(self):
        self.stdDev = 0.35


    def init_weights(self, shape):
        return tf.Variable(tf.random_normal(shape, stddev=self.stdDev))


    def conv_layer(self, prev_layer_out, w_shape, layer_stride, w_name, num_dim = '2d', if_relu = True, batchNorm = True):
        w_conv = tf.Variable(tf.random_normal(w_shape, stddev=self.stdDev),
                          name=w_name)
        
        numFilters = w_shape[len(w_shape)-1]
        b = tf.Variable(tf.random_normal(numFilters, stddev=self.stdDev))

        nextLayer = None
        if num_dim == '3d':
            nextLayer = tf.add(tf.nn.conv3d(prev_layer_out, w_conv, 
                            strides=layer_stride, padding='SAME',use_cudnn_on_gpu=False,name=w_name),b)
        else:
            nextLayer = tf.add(tf.nn.conv2d(prev_layer_out, w_conv, 
                            strides=layer_stride, padding='SAME',use_cudnn_on_gpu=False,name=w_name),b)

        if batchNorm:
            nextLayer = batch_norm(nextLayer, [numFilters], [numFilters], [numFilters], [numFilters])

        if if_relu:
            nextLayer = relu(nextLayer)

        
        return nextLayer, w_conv


    def deconv_layer(self, prev_layer_out, filter_shape, out_shape, strides, padding='SAME', num_dim = '2d'):
        w_deconv = tf.Variable(tf.random_normal(w_shape, stddev=self.stdDev),
                          name=w_name)

        nextLayer = None
        if num_dim == '3d':
            nextLayer = tf.nn.conv3d_transpose(prev_layer_out, [], 
                            strides=layer_stride, padding='SAME',use_cudnn_on_gpu=False,name=w_name)
        else:
            nextLayer = tf.nn.conv2d_transpose(prev_layer_out, w_conv, 
                            strides=layer_stride, padding='SAME',use_cudnn_on_gpu=False,name=w_name)
        
        return nextLayer, w_deconv
        
    def pool(self, prev_layer, window_size, str_size, poolType = 'max'):
        next_layer = None
        if poolType == 'max':
            next_layer = tf.nn.max_pool3d(prev_layer, ksize=window_size,
                            strides=str_size, padding='SAME')
        if poolType == 'avg':
            next_layer = tf.nn.avg_pool3d(prev_layer, ksize=window_size,
                            strides=str_size, padding='SAME')

        return next_layer

    def dropout(self, prev_layer,  p_keep):
        next_layer = tf.nn.dropout(prev_layer, p_keep)
        return next_layer

    def batch_norm(self, prev_layer, mu_shape, sig_shape, beta_shape,scale_shape, var_eps = 1e-6):
        mu = init_weights(mu_shape)
        sigma = init_weights(sig_shape)
        beta = init_weights(beta_shape)
        scale = init_weights(scale_shape)
        next_layer = tf.nn.batch_normalization(prev_layer, mu, sigma, beta, scale, var_eps)
        return next_layer

    def relu(self, prev_layer):
        next_layer = tf.nn.relu(prev_layer)
        return next_layer

    def lrelu(x, leak=0.2, name="lrelu"):
        return tf.maximum(x, leak*x)

    def fcLayer(self, prev_layer, shape):
        wOut = init_weights(shape)
        pyx = tf.matmul(layer3, wOut)
        return pyx


    def createVariables(self, x_shape, y_shape):
        # X = tf.placeholder("float", x_shape)
        # Y = tf.placeholder("float", y_shape)
        p_keep_conv = tf.placeholder("float")
        # return X,Y,p_keep_conv

        input_x, input_y = inputs(True, '/data/brain_binary_list.json', 2)

        return input_x, input_y, p_keep_conv

    def cost_function(self, model_output, Y):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model_output, Y))
        return cost

    def minimization_function(self, cost, learning_rate, beta1, beta2):
        train_op = tf.train.RMSPropOptimizer(learning_rate, beta1).minimize(cost)
        return train_op

    def prediction(self, model_output):
        predict_op = tf.argmax(model_output, 1)
        return predict_op


    def build_simple_model(self, X, Y):
         layer1, w_1 = self.conv_layer( X, [3, 3, 3, 1, 16], [1, 1, 1, 1], "layer1_filters", '3d', True, True)
         layer3, w_3 = self.conv_layer( layer1, [3, 3, 3, 16, 16], [1, 1, 1, 1], "layer2_filters", '3d', True, True)
         layer3 = self.pool(layer3,[1, 5, 9, 5, 3],[1, 5, 9, 5, 3] , 'max')
         layer3 = tf.reshape(layer3, [1, 810])
         pyx = self.fcLayer(layer3, [ 810, 1])
         return pyx

    def cnn_autoencoder(self, X, input):
        encode = []
        decode = []
        layer1, w_1 = self.conv_layer( X, [5, 5, 5, 1, 10], [1, 1, 1, 1], "layer1_filters", '3d', True)
        layer1 = self.pool(layer1,[1, 5, 9, 5, 3],[1, 5, 9, 5, 3] , 'avg')
        encode.append(layer1)
        layer3, w_3 = self.deconv_layer(layer1, [2, 2, 2, 10, 1], [1, 45, 54, 45], [1, 1, 1, 1], padding='SAME', num_dim='3d')
        decode.append(layer3)

        return encode, decode




