import tensorflow as tf
import numpy as np
from input_brain import *
from utils import *


# Define custom API for creating and adding layers to NN Model
# Wrapper around Tensorflow API, for ease of use and readibility

class Layers(object):

    def __init__(self):
        self.stdDev = 0.35

    ''' Initializes the weights based on the std dev set in the constructor
    
    '''
    def init_weights(self, shape):
        return tf.Variable(tf.random_normal(shape, stddev=self.stdDev))

    def createVariables(self, x_shape, y_shape, batch_size):
        # X = tf.placeholder("float", x_shape)
        # Y = tf.placeholder("float", y_shape)
        p_keep_conv = tf.placeholder("float")
        # return X,Y,p_keep_conv

        X, Y = inputs(True, '/data/train_brain_binary_list.json', batch_size)
        return X,Y,p_keep_conv


    def dropout(self, prev_layer,  p_keep):
        next_layer = tf.nn.dropout(prev_layer, p_keep)
        return next_layer

    def batch_norm(self, prev_layer, mu_shape, sig_shape, beta_shape,scale_shape, var_eps = 1e-3):
        mu = self.init_weights(mu_shape)
        sigma = self.init_weights(sig_shape)
        beta = self.init_weights(beta_shape)
        scale = self.init_weights(scale_shape)
        next_layer = tf.nn.batch_normalization(prev_layer, mu, sigma, beta, scale, var_eps)
        return next_layer

    def fcLayer(self, prev_layer, shape):
        wOut = self.init_weights(shape)
        pyx = tf.matmul(prev_layer, wOut)
        return pyx

    def cost_function(self, model_output, Y):
        # print("The shape of the model output is: " + str(model_output.get_shape()))
        cost = tf.reduce_mean(tf.square(model_output - Y))
        return cost

    def minimization_function(self, cost, learning_rate, beta1, beta2, opt='Rmsprop'):
        train_op = None
        if opt == 'Rmsprop':
            train_op = tf.train.RMSPropOptimizer(learning_rate, beta1).minimize(cost)
        elif opt == 'adam':
            train_op = tf.train.AdamOptimizer(learning_rate, beta1, beta2).minimize(cost)
        elif opt == 'adagrad':
            train_op = tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value=0.1).minimize(cost)

        return train_op

    def add_regularization(self, loss, wgt, lmbda, rho, op='kl'):
        nextLoss = None
        if op == 'l2':
            nextLoss = tf.add(loss, lmbda*tf.nn.l2_loss(wgt))
        elif op == 'kl':
            nextLoss = tf.add(loss, tf.mul(lmbda, self.kl_sparse_regularization(wgt, lmbda, rho)))
        return nextLoss

    def kl_sparse_regularization(self, wgt, lmbda, rho):
        rho_hat = tf.reduce_mean(wgt)
        invrho = tf.sub(tf.constant(1.), rho)
        invrhohat = tf.sub(tf.constant(1.), rho_hat)
        logrho = tf.add(self.logfunc(rho,rho_hat), self.logfunc(invrho, invrhohat))
        return logrho

    def logfunc(self, x1, x2):
        clippDiv = tf.clip_by_value(tf.div(x1,x2),1e-12,1e10)
        return tf.mul( x1,tf.log(clippDiv))


    def prediction(self, model_output):
        predict_op = tf.argmax(model_output, 1)
        return predict_op


class CNNLayers(Layers):

    ''' Constructor for the ConvolutionalNN class. Initializes the
    std dev for the distributions used for weight initializations
    '''
    def __init__(self):
        Layers.__init__(self)
        self.stdDev = 0.35

    def init_weights(self, shape):
        return tf.Variable(tf.random_normal(shape, stddev=self.stdDev))

    def conv_layer(self, prev_layer_out, w_shape, layer_stride, w_name, num_dim = '2d', padding='SAME',if_relu = True, batchNorm = True):
        w_conv = tf.Variable(tf.random_normal(w_shape, stddev=self.stdDev),
                          name=w_name)
        
        numFilters = w_shape[len(w_shape)-1]
        b = tf.Variable(tf.random_normal([numFilters], stddev=self.stdDev))

        nextLayer = None
        if num_dim == '3d':
            nextLayer = tf.add(tf.nn.conv3d(prev_layer_out, w_conv, 
                            strides=layer_stride, padding=padding,name=w_name),b)
        else:
            nextLayer = tf.add(tf.nn.conv2d(prev_layer_out, w_conv, 
                            strides=layer_stride, padding=padding,name=w_name),b)

        if batchNorm:
            nextLayer = self.batch_norm(nextLayer, [numFilters], [numFilters], [numFilters], [numFilters])

        if if_relu:
            nextLayer = self.relu(nextLayer)

        
        return nextLayer, w_conv


    def deconv_layer(self, prev_layer_out, filter_shape, out_shape, layer_stride, w_name, num_dim = '2d',padding='SAME', if_relu = True, batchNorm = True):
        w_deconv = tf.Variable(tf.random_normal(filter_shape, stddev=self.stdDev),
                          name=w_name)


        numFilters =filter_shape[len(filter_shape)-1]
        b = tf.Variable(tf.random_normal([numFilters], stddev=self.stdDev))

        nextLayer = None

        if num_dim == '3d':
            nextLayer = tf.add(tf.nn.conv3d_transpose(prev_layer_out, w_deconv, out_shape, 
                            strides=layer_stride, padding=padding),b)
        else:
            nextLayer = tf.add(tf.nn.conv2d_transpose(prev_layer_out, w_deconv, out_shape, 
                            strides=layer_stride, padding=padding),b)

        if batchNorm:
            nextLayer = self.batch_norm(nextLayer, [numFilters], [numFilters], [numFilters], [numFilters])

        if if_relu:
            nextLayer = self.relu(nextLayer)

        
        return nextLayer, w_deconv
        
    def pool(self, prev_layer, window_size, str_size, poolType = 'max'):
        next_layer = None
        if poolType == 'max':
            next_layer = tf.nn.max_pool3d(prev_layer, ksize=window_size,
                            strides=str_size, padding='SAME')
        elif poolType == 'avg':
            next_layer = tf.nn.avg_pool3d(prev_layer, ksize=window_size,
                            strides=str_size, padding='SAME')

        return next_layer

    def relu(self, prev_layer):
        next_layer = tf.nn.relu(prev_layer)
        return next_layer

    def lrelu(x, leak=0.2, name="lrelu"):
        return tf.maximum(x, leak*x)


    def residual_unit(self, input_layer, output_layer):
        res = input_layer + output_layer
        return res

    def simple_cnn_model(self, X, batch_size):
        wList = []
        layer1, w_1 = self.conv_layer( X, [3, 3, 3, 1, 1], [1, 1, 1, 1, 1], "layer1_filters", '3d', True, True)
        layer2, w_3 = self.conv_layer( layer1, [3, 3, 3, 1, 1], [1, 1, 1, 1, 1], "layer2_filters", '3d', True, True)
        layer2 = self.pool(layer2,[1, 5, 9, 5, 1],[1, 5, 9, 5, 1] , 'max')
        layer2 = tf.reshape(layer2, [1, 15552])
        pyx = self.fcLayer(layer2, [ 15552, batch_size])
        wList.append(w_1)
        wList.append(w_3)
        return pyx, wList

    def cnn_autoencoder(self, X, batch_size):
        ''' Have lists to store the outputs of the encoding (dimension reduction) layer and
            decoding (reconstruction) layer
        '''

        weightList = []
        encode = []
        decode = []

        ''' Build the Convolutional AutoEncoder, without pooling and unpooling. Tensorflow, as of now, does not 
            support unpooling.
        '''
        layer1, w_1 = self.conv_layer( X, [3, 3, 3, 1, 1], [1, 1, 1, 1, 1], "layer1_filters", '3d', 'SAME', True, True)
        layer2, w_2 = self.conv_layer( layer1, [3, 3, 3, 1, 1], [1, 1, 1, 1, 1], "layer2_filters", '3d', 'SAME', True, True)
        layer3, w_3 = self.conv_layer( layer2, [3, 3, 3, 1, 1], [1, 1, 1, 1, 1], "layer3_filters", '3d', 'SAME', True, True)
        layer4, w_4 = self.conv_layer( layer3, [3, 3, 3, 1, 1], [1, 2, 2, 2, 1], "layer4_filters", '3d', 'SAME', True, True)
        layer5, w_5 = self.conv_layer( layer4, [3, 3, 3, 1, 1], [1, 2, 2, 2, 1], "layer5_filters", '3d', 'SAME', True, True)
        encode.append(layer5)
        weightList.append(w_1)
        weightList.append(w_2)
        weightList.append(w_3)
        weightList.append(w_4)
        weightList.append(w_5)

        l4Shape = layer4.get_shape().as_list()
        l3Shape = layer3.get_shape().as_list()
        l2Shape = layer2.get_shape().as_list()
        l1Shape = layer1.get_shape().as_list()
        XShape = X.get_shape().as_list()

        print("This is the lowest dimension shape: " + str(layer5.get_shape().as_list()))

        layer6, w_6 = self.deconv_layer(layer5, [3, 3, 3, 1, 1], l4Shape, [1, 2, 2, 2, 1], 'layer6_filters', '3d', 'SAME', True, True)
        layer7, w_7 = self.deconv_layer(layer6, [3, 3, 3, 1, 1], l3Shape, [1, 2, 2, 2, 1], 'layer7_filters', '3d', 'SAME', True, True)
        layer8, w_8 = self.deconv_layer(layer7, [3, 3, 3, 1, 1], l2Shape, [1, 1, 1, 1, 1], 'layer8_filters', '3d', 'SAME', True, True)
        layer9, w_9 = self.deconv_layer(layer8, [3, 3, 3, 1, 1], l1Shape, [1, 1, 1, 1, 1], 'layer9_filters', '3d', 'SAME', True, True)
        layer10, w_10 = self.deconv_layer(layer9, [3, 3, 3, 1, 1], XShape, [1, 1, 1, 1, 1], 'layer10_filters', '3d', 'SAME', True, True)

        weightList.append(w_6)
        weightList.append(w_7)
        weightList.append(w_8)
        weightList.append(w_9)
        weightList.append(w_10)

        decode.append(layer10)
        # autoSess.close()

        return encode, decode, weightList




