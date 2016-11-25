import tensorflow as tf
import numpy as np
from utils import *
from layers import *
from input_brain import *
import copy

class ConvAutoEncoder(NeuralNetwork):

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
            print(filters[tot-i-1])
            layer_outputs['layer'+str(i+1)] ,weights['w'+str(i+1)] = self.deconv_layer(prev_layer,filters[tot-i-1], layer_shapes['w'+str(tot-i-1)], strides[tot-i-1], names[i], '3d', 'SAME', relu[i], batch_norm[i])
            prev_layer = layer_outputs['layer'+str(i+1)]
            layer_shapes['w'+str(i+1)] = prev_layer.get_shape().as_list()

        print("The decoded image size is: " + str(prev_layer.get_shape().as_list()))

        decode.append(prev_layer)
        self.decode = prev_layer
        self.layer_outputs = layer_outputs
        self.weights = weights

        return layer_outputs, weights, layer_shapes, encode, decode

    def train(self):
        cost = self.cost_function(self.decode, self.input, op='square')
        cumCost = self.add_regularization( cost, self.encode, self.lmbda, self.rho, op='kl')
        train_op = self.minimization_function(cumCost, self.learning_rate, self.beta1, self.beta2, self.op)
        return cumCost, train_op


class ConvNN(CNNLayers):
    def __init__(self, input_shape, output_shape, num_layers, batch_size=1, learning_rate=1e-3, beta1=0.99, beta2=0.99, lmbda = [0]*num_layers, op='Rmsprop'):
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

    def build_model(self):
        weights = {}
        layersOut = {}

        layersOut['layer1'], weights['w1'] = self.conv_layer( X, [3, 3, 3, 1, 1], [1, 1, 1, 1, 1], 'layer1_filters', '3d', True, True)
        layersOut['layer2'], weights['w2'] = self.conv_layer(layersOut['layer1'], [3, 3, 3, 1, 1], [1, 1, 1, 1, 1], 'layer2_filters', '3d', True, True)

        layersOut['layer2-pool'] = self.pool(layersOut['layer2'],[1, 3, 3, 3, 1],[1, 1, 1, 1, 1] , 'max')

        layersOut['layer3'], weights['w3']=  self.conv_layer( layersOut['layer2-pool'], [3, 3, 3, 1, 1], [1, 1, 1, 1, 1], 'layer3_filters', '3d', True, True)
        layersOut['layer4'], weights['w4']=  self.conv_layer( layersOut['layer3'], [3, 3, 3, 1, 1], [1, 1, 1, 1, 1], 'layer4_filters', '3d', True, True)

        layersOut['layer4-pool'] = self.pool(layersOut['layer4'],[1, 5, 5, 5, 1],[1, 2, 2, 2, 1] , 'max')

        layersOut['layer5'], weights['w5'] =  self.conv_layer( layersOut['layer4-pool'], [3, 3, 3, 1, 1], [1, 1, 1, 1, 1], 'layer5_filters', '3d', True, True)

        fcShapeConv = weights['w5'].get_shape().as_list()
        numParams = reduce(lambda x, y: x*y, fcShapeConv)
        layersOut['layer5-fc'] = tf.reshape(layersOut['layer5'], [1, numParams])
        layersOut['layer6'], weights['w6'] = self.fcLayer(layersOut['layer5-fc'], [ numParams, numParams])
        layersOut['pred'], weights['w7'] = self.fcLayer(layersOut['layer6'], [ numParams, self.batch_size])
        layersOut['pred-prob'] = self.sigmoid(layersOut['pred'])


        self.layersOut = layersOut
        self.weights = weights

        return layersOut, weights

    def train(self):
        cost = self.cost_function( self.layersOut['pred-prob'], self.output, op='square')
        cumCost = cost
        numEntries = len(self.weights)
        for i in range(numEntries):
            cumCost = self.add_regularization( cumCost, self.weights[i], self.lmbda[i], None, op='l2')

        train_op = self.minimization_function(cumCost, self.learning_rate, self.beta1, self.beta2, self.op)
        return cumCost, train_op



# class ResCNN(CNNLayers):
#     def __init__(self, input_shape, output_shape, batch_size=1, learning_rate=1e-3, beta1=0.99, beta2=0.99, rho=0.4, lmbda = 0.6, op='Rmsprop'):
#         CNNLayers.__init__(self)
#         self.input, self.output, self.p_keep = self.createVariables(input_shape, output_shape, batch_size)
#         self.input_shape = input_shape
#         self.output_shape = output_shape
#         self.batch_size = batch_size
#         self.learning_rate = learning_rate
#         self.beta1 = beta1
#         self.beta2 = beta2
#         self.rho = rho
#         self.lmbda = lmbda
#         self.op = op


#     def build_model(self):







