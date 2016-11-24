import tensorflow as tf
import numpy as np
from utils import *
from ConvolutionalNN import *
from input_brain import *
import copy


class ConvAutoEncoder(CNNLayers):

    def __init__(self, input_shape, output_shape, batch_size=1, learning_rate=1e-3, beta1=0.99, beta2=0.99, op='Rmsprop'):
        CNNLayers.__init__(self)
        self.input, self.output, self.p_keep = self.createVariables(input_shape, output_shape, batch_size)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.op = op


    def build_model(self, num_layers_encode, filters, strides, names, relu, batch_norm):
        weights = {}
        layer_outputs = {}
        weight_shapes = {}
        encode = []
        decode = []

        prev_layer = self.input
        weight_shapes['w0'] =  self.input.get_shape().as_list()
        for i in range(num_layers_encode):
            print("This is the index of layer: " + str(i))
            layer_outputs['layer'+str(i+1)] ,weights['w'+str(i+1)] = self.conv_layer( prev_layer, filters[i], strides[i], names[i], '3d', 'SAME', relu[i], batch_norm[i])
            prev_layer = layer_outputs['layer'+str(i+1)]
            weight_shapes['w'+str(i+1)] = weights['w'+str(i+1)].get_shape().as_list()

        encode.append(prev_layer)
        self.encode = prev_layer

        # print("These are the layers added: " + str(layer_outputs))
        
        tot = 2*num_layers_encode
        for i in range(num_layers_encode, tot):
            print("This is the indexing for the reverse filters and weight shapes: " + str(i))
            layer_outputs['layer'+str(i+1)] ,weights['w'+str(i+1)] = self.deconv_layer(layer_outputs['layer'+str(i)],filters[tot-i-1], weight_shapes['w'+str(tot-i-1)], strides[tot-i-1], names[i], '3d', 'SAME', relu[i], batch_norm[i])
            prev_layer = layer_outputs['layer'+str(i+1)]
            weight_shapes['w'+str(i+1)] = weights['w'+str(i+1)].get_shape().as_list()


        # print("These are the layers added: " + str(weight_shapes))
        decode.append(prev_layer)
        self.decode = prev_layer

        return layer_outputs, weights, weight_shapes, encode, decode

    def train(self):
        cost = self.cost_function(self.decode, self.input)
        train_op = self.minimization_function(cost, self.learning_rate, self.beta1, self.beta2, self.op)
        return cost, train_op


# class ConvNN(CNNLayers):


# class ResCNN(CNNLayers):








