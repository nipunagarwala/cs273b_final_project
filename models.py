import tensorflow as tf
import numpy as np
from utils import *
from ConvolutionalNN import *
from input_brain import *
import copy


class ConvAutoEncoder(CNNLayers):

    def __init__(self, input_shape, output_shape, batch_size=1, learning_rate=1e-3, beta1=0.99, beta2=0.99, rho=0.4, lmbda = 0.6, op='Rmsprop'):
        CNNLayers.__init__(self)
        self.input, self.output, self.p_keep = self.createVariables(input_shape, output_shape, batch_size)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.rho = rho
        self.lmbda = lmbda
        self.op = op


    def build_model(self, num_layers_encode, filters, strides, names, relu, batch_norm):
        weights = {}
        layer_outputs = {}
        layer_shapes = {}
        encode = []
        decode = []

        prev_layer = self.input
        layer_shapes['w0'] =  self.input.get_shape().as_list()
        for i in range(num_layers_encode):
            layer_outputs['layer'+str(i+1)] ,weights['w'+str(i+1)] = self.conv_layer( prev_layer, filters[i], strides[i], names[i], '3d', 'SAME', relu[i], batch_norm[i])
            prev_layer = layer_outputs['layer'+str(i+1)]
            layer_shapes['w'+str(i+1)] = prev_layer.get_shape().as_list()

        encode.append(prev_layer)
        self.encode = prev_layer

        print("The encoded image size is: " + str(prev_layer.get_shape().as_list()))
        
        tot = 2*num_layers_encode #4 layers
        for i in range(num_layers_encode, tot):
            layer_outputs['layer'+str(i+1)] ,weights['w'+str(i+1)] = self.deconv_layer(prev_layer,filters[tot-i-1], layer_shapes['w'+str(tot-i-1)], strides[tot-i-1], names[i], '3d', 'SAME', relu[i], batch_norm[i])
            prev_layer = layer_outputs['layer'+str(i+1)]
            layer_shapes['w'+str(i+1)] = prev_layer.get_shape().as_list()


        decode.append(prev_layer)
        self.decode = prev_layer
        self.layer_outputs = layer_outputs
        self.weights = weights

        return layer_outputs, weights, layer_shapes, encode, decode

    def train(self):
        cost = self.cost_function(self.decode, self.input)
        cumCost = self.add_regularization( cost, self.encode, self.lmbda, self.rho, op='kl')
        train_op = self.minimization_function(cumCost, self.learning_rate, self.beta1, self.beta2, self.op)
        return cumCost, train_op


# class ConvNN(CNNLayers):


# class ResCNN(CNNLayers):








