import tensorflow as tf
import numpy as np
from utils import *
from layers import *
from input_brain import *
import copy


class NeuralNetwork(CNNLayers):
    def __init__(self, data, output, p_keep_conv, batch_size=1, learning_rate=1e-3, beta1=0.99, beta2=0.99, w_lmbda=None, b_lmbda = None , op='Rmsprop'):
        CNNLayers.__init__(self)
        self.input_data = data
        self.input_data = tf.reshape(self.input_data, [batch_size,29 ], name=None)

        self.output = output
        self.dropout = p_keep_conv
        self.output = tf.reshape(self.output, [batch_size,1 ], name=None)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.w_lmbda = w_lmbda
        self.b_lmbda = b_lmbda
        self.op = op

    def build_model(self, num_layers, hidden_units, sigmoid=True, batch_norm=True, in_multiModal=True):
        weights = {}
        layersOut = {}
        biases = {}

        layersOut['input'] = self.input_data
        layersOut['output'] = self.output
        prev_layer = self.input_data

        prev_shape = (prev_layer.get_shape().as_list())[1]
        for i in range(num_layers-1):
            layersOut['layer'+str(i+1)] ,weights['w'+str(i+1)], biases['w'+str(i+1)] = self.fcLayer(prev_layer, [prev_shape, hidden_units[i]], sigmoid, batch_norm)
            prev_shape = hidden_units[i]
            prev_layer = layersOut['layer'+str(i+1)]

        layersOut['layer'+str(num_layers)] ,weights['w'+str(num_layers)], biases['w'+str(num_layers)] = self.fcLayer(prev_layer, [prev_shape, hidden_units[num_layers-1]], False, False)
        layersOut['output_values'] = layersOut['layer'+str(num_layers)]
        layersOut['pred'] = tf.nn.softmax(layersOut['output_values'], dim=-1, name=None)
        layersOut['fc-mmnn'] = layersOut['layer'+str(in_multiModal)]

        self.layersOut = layersOut
        self.weights = weights
        self.biases = biases

        return layersOut, weights

    def train(self, nn_reg_on, nn_reg_op):
        cost = self.cost_function(self.layersOut['output_values'], self.output, op='log-likelihood')
        cumCost = cost
        numEntries = len(self.weights)

        if nn_reg_on:
            weightVals = self.weights.values()
            biasVals = self.biases.values()
            for i in range(numEntries):
                cumCost = self.add_regularization( cumCost, weightVals[i], self.w_lmbda [i], None, op= nn_reg_op)
                cumCost = self.add_regularization( cumCost, biasVals[i], self.b_lmbda [i], None, op= nn_reg_op)

        train_op = self.minimization_function(cumCost, self.learning_rate, self.beta1, self.beta2, self.op)
        return cumCost, train_op


class AutoEncoder(CNNLayers):

    def __init__(self, data, output, p_keep_conv, batch_size=1, learning_rate=1e-3, beta1=0.99, beta2=0.99, rho=0.4, lmbda = 0.6, op='Rmsprop'):
        Layers.__init__(self)
        self.input_data = data
        self.input_data = tf.reshape(self.input_data, [batch_size,29], name=None)

        self.output = output
        self.dropout = p_keep_conv

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.lmbda = lmbda
        self.rho = rho
        self.op = op


    def build_model(self, hidden_units, relu, batch_norm):
        weights = {}
        biases = {}
        layer_outputs = {}
        layer_shapes = {}
        encode = []
        decode = []

        layer_outputs['input'] = self.input_data
        layer_outputs['output'] = self.output
        prev_layer = self.input_data

        unit_sizes = [(prev_layer.get_shape().as_list())[1]]
        unit_sizes += hidden_units

        for i in range(len(hidden_units)):
            layer_outputs['layer'+str(i+1)], weights['w'+str(i+1)], biases['w'+str(i+1)] \
                    = self.fcLayer(prev_layer, unit_sizes[i:i+2], relu[i], batch_norm[i])
            prev_layer = layer_outputs['layer'+str(i+1)]
            print unit_sizes[i:i+2]

        encode.append(prev_layer)
        self.encode = prev_layer

        print("The encoded data size is: " + str(prev_layer.get_shape().as_list()))

        unit_sizes.reverse()
        tot = 2*len(hidden_units)
        for i in range(len(hidden_units), tot):
            layer_outputs['layer'+str(i+1)], weights['w'+str(i+1)], biases['w'+str(i+1)] \
                    = self.fcLayer(prev_layer, unit_sizes[i-len(hidden_units):i-len(hidden_units)+2],
                                   relu[i], batch_norm[i])
            prev_layer = layer_outputs['layer'+str(i+1)]
            print unit_sizes[i-len(hidden_units):i-len(hidden_units)+2]

        print("The decoded data size is: " + str(prev_layer.get_shape().as_list()))

        decode.append(prev_layer)
        self.decode = prev_layer
        self.layer_outputs = layer_outputs
        self.weights = weights
        self.biases = biases

        return layer_outputs, weights, layer_shapes, encode, decode, self.input_data, self.output

    def train(self):
        cost = self.cost_function(self.decode, self.input_data, op='softmax')

        cumCost = cost
        numEntries = len(self.weights)

        weightVals = self.weights.values()
        # for i in range(numEntries):
        #     cumCost = self.add_regularization( cumCost, weightVals[i], self.lmbda[i], self.rho[i], op='kl')

        cumCost = self.add_regularization( cost, self.encode, self.lmbda, self.rho, op='kl')
        train_op = self.minimization_function(cumCost, self.learning_rate, self.beta1, self.beta2, self.op)
        return cumCost, train_op



class ConvAutoEncoder(CNNLayers):

    def __init__(self, image, output, p_keep_conv, batch_size=1, learning_rate=1e-3, beta1=0.99, beta2=0.99, rho=0.4, lmbda = 0.6, op='Rmsprop'):
        CNNLayers.__init__(self)
        self.input_image = image
        self.output = output
        self.dropout = p_keep_conv

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.lmbda = lmbda
        self.rho = rho
        self.op = op


    def build_model(self, num_layers_encode, filters, strides, names, relu, batch_norm):
        weights = {}
        biases = {}
        layer_outputs = {}
        layer_shapes = {}
        encode = []
        decode = []

        layer_outputs['input'] = self.input_image
        layer_outputs['output'] = self.output
        prev_layer = self.input_image
        layer_shapes['w0'] =  self.input_image.get_shape().as_list()

        for i in range(num_layers_encode):
            layer_outputs['layer'+str(i+1)] ,weights['w'+str(i+1)], biases['b'+str(i+1)] = self.conv_layer( prev_layer, filters[i], strides[i], names[i], '3d', 'SAME', relu[i], batch_norm[i])
            prev_layer = layer_outputs['layer'+str(i+1)]
            layer_shapes['w'+str(i+1)] = prev_layer.get_shape().as_list()

        encode.append(prev_layer)
        self.encode = prev_layer

        print("The encoded image size is: " + str(prev_layer.get_shape().as_list()))

        tot = 2*num_layers_encode #4 layers
        for i in range(num_layers_encode, tot):
            print(filters[tot-i-1])
            layer_outputs['layer'+str(i+1)] ,weights['w'+str(i+1)],biases['b'+str(i+1)] = self.deconv_layer(prev_layer,filters[tot-i-1], layer_shapes['w'+str(tot-i-1)], strides[tot-i-1], names[i], '3d', 'SAME', relu[i], batch_norm[i])
            prev_layer = layer_outputs['layer'+str(i+1)]
            layer_shapes['w'+str(i+1)] = prev_layer.get_shape().as_list()

        print("The decoded image size is: " + str(prev_layer.get_shape().as_list()))

        decode.append(prev_layer)
        self.decode = prev_layer
        self.layer_outputs = layer_outputs
        self.weights = weights

        return layer_outputs, weights, encode, decode, self.input_image

    def train(self):
        cost = self.cost_function(self.decode, self.input_image, op='softmax')

        cumCost = cost
        numEntries = len(self.weights)

        weightVals = self.weights.values()
        # for i in range(numEntries):
        #     cumCost = self.add_regularization( cumCost, weightVals[i], self.lmbda[i], self.rho[i], op='kl')

        cumCost = self.add_regularization( cost, self.encode, self.lmbda, self.rho, op='kl')
        train_op = self.minimization_function(cumCost, self.learning_rate, self.beta1, self.beta2, self.op)
        return cumCost, train_op


class ConvNN(CNNLayers):
    def __init__(self, image, output, p_keep_conv, num_layers, batch_size=1, learning_rate=1e-3, beta1=0.99, beta2=0.99, w_lmbda = None,b_lmbda = None, op='Rmsprop'):
        CNNLayers.__init__(self)
        self.input_image = image
        self.output = output
        self.output = tf.reshape(self.output, [batch_size,1 ], name=None)
        self.dropout = p_keep_conv
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.b_lmbda = b_lmbda
        self.w_lmbda = w_lmbda
        self.op = op
        # self.output = tf.reshape(self.output, [1,self.batch_size], name=None)

    # def build_model(self, sigmoid, batch_norm):
    #     weights = {}
    #     layersOut = {}

    #     layersOut['input'] = self.input_image
    #     layersOut['output'] = self.output

    #     layersOut['layer1'], weights['w1'] = self.conv_layer( self.input_image, [3, 3, 3, 1, 1], [1, 1, 1, 1, 1], 'layer1_filters', '3d','SAME', True, True)

    #     layersOut['layer2'], weights['w2'] = self.conv_layer(layersOut['layer1'], [3, 3, 3, 1, 1], [1, 1, 1, 1, 1], 'layer2_filters', '3d', 'SAME',True, True)

    #     layersOut['layer2-pool'] = self.pool(layersOut['layer2'],[1, 5, 5, 5, 1],[1, 2, 2, 2, 1] , 'max')

    #     layersOut['layer3'], weights['w3']=  self.conv_layer( layersOut['layer2-pool'], [3, 3, 3, 1, 1], [1, 1, 1, 1, 1], 'layer3_filters', '3d','SAME', True, True)
    #     layersOut['layer4'], weights['w4']=  self.conv_layer( layersOut['layer3'], [3, 3, 3, 1, 1], [1, 1, 1, 1, 1], 'layer4_filters', '3d', 'SAME',True, True)

    #     layersOut['layer4-pool'] = self.pool(layersOut['layer4'],[1, 5, 5, 5, 1],[1, 2, 2, 2, 1] , 'max')

    #     layersOut['layer5'], weights['w5'] =  self.conv_layer( layersOut['layer4-pool'], [3, 3, 3, 1, 1], [1, 1, 1, 1, 1], 'layer5_filters', '3d','SAME', True, True)
    #     layersOut['layer6'], weights['w6']=  self.conv_layer( layersOut['layer5'], [3, 3, 3, 1, 1], [1, 1, 1, 1, 1], 'layer6_filters', '3d', 'SAME',True, True)

    #     layersOut['layer6-pool'] = self.pool(layersOut['layer6'],[1, 5, 5, 5, 1],[1, 2, 2, 2, 1] , 'max')

    #     fcShapeConv = layersOut['layer6-pool'].get_shape().as_list()
    #     fcShapeConv = fcShapeConv[1:]
    #     numParams = reduce(lambda x, y: x*y, fcShapeConv)

    #     layersOut['layer6-fc'] = tf.reshape(layersOut['layer6-pool'], [self.batch_size, numParams])
    #     layersOut['layer7'], weights['w7'] = self.fcLayer(layersOut['layer6-fc'], [ numParams, numParams], sigmoid, batch_norm)
    #     layersOut['pred'], weights['w8'] = self.fcLayer(layersOut['layer7'], [ numParams, 1], True, True)

    #     layersOut['pred'] = tf.reshape(layersOut['pred'], [1,self.batch_size], name=None)

    #     layersOut['cnn_out'] = layersOut['layer6-fc']


    #     self.layersOut = layersOut
    #     self.weights = weights

    #     return layersOut, weights

    def build_model(self, conv_arch, cnn_num_layers, cnn_num_fc_layers,
                        cnn_filter_sz, cnn_num_filters, cnn_stride_sz, cnn_pool_sz, cnn_pool_stride_sz, cnn_batch_norm):

        weights = {}
        layersOut = {}
        biases = {}

        layersOut['input'] = self.input_image
        layersOut['output'] = self.output

        prev_layer = self.input_image

        prev_layer_fltr = 1
        layer_counter = 0
        num_fc_done = 0
        while True:
            i = layer_counter
            ftlr_sz = cnn_filter_sz[i]
            ftlr_str_sz = cnn_stride_sz[i]
            pool_win_sz = cnn_pool_sz[i]
            pool_str_sz = cnn_pool_stride_sz[i]
            layer_num_ftlr = cnn_num_filters[i]

            print("This is the current counter: " + str(layer_counter))
            if conv_arch[i] == 'conv':
                w_name = 'layer'+str(i+1)+'_filters'
                layersOut['layer'+str(i+1)] ,weights['w'+str(i+1)], biases['b'+str(i+1)]= self.conv_layer(prev_layer, [ftlr_sz, ftlr_sz, ftlr_sz, prev_layer_fltr ,layer_num_ftlr ],
                                     [1 , ftlr_str_sz , ftlr_str_sz , ftlr_str_sz, 1], w_name, '3d','SAME', True, cnn_batch_norm)

                prev_layer_fltr = layer_num_ftlr
                print("This is the Convolutional Layer")
                print("This is the shape of the outputs of this layer: " + str(layersOut['layer'+str(i+1)].get_shape().as_list()))

            if conv_arch[i] == 'maxpool':
                layersOut['layer'+str(i+1)] = self.pool(prev_layer,[1, pool_win_sz, pool_win_sz, pool_win_sz, 1],
                                                [1, pool_str_sz, pool_str_sz, pool_str_sz, 1] , 'max')
                print("This is the Max Pool Layer")
                print("This is the shape of the outputs of this layer: " + str(layersOut['layer'+str(i+1)].get_shape().as_list()))

            if conv_arch[i] == 'avgpool':
                layersOut['layer'+str(i+1)] = self.pool(prev_layer,[1, pool_win_sz, pool_win_sz, pool_win_sz, 1],
                                                [1, pool_str_sz, pool_str_sz, pool_str_sz, 1] , 'avg')
                print("This is the Average Pool Layer")
                print("This is the shape of the outputs of this layer: " + str(layersOut['layer'+str(i+1)].get_shape().as_list()))

            if conv_arch[i] == 'reshape':
                fcShapeConv = prev_layer.get_shape().as_list()
                fcShapeConv = fcShapeConv[1:]
                self.cnn_out_params = reduce(lambda x, y: x*y, fcShapeConv)
                layersOut['layer'+str(i+1)+'-fc'] = tf.reshape( prev_layer, [-1, self.cnn_out_params])
                layersOut['cnn_out'] = layersOut['layer'+str(i+1)+'-fc']
                prev_layer = layersOut['layer'+str(i+1)+'-fc']
                layer_counter += 1
                print("This is the FC Reshape Layer")
                print("This is the shape of the outputs of this layer: " + str(layersOut['layer'+str(i+1)+'-fc'].get_shape().as_list()))
                print()
                continue

            if conv_arch[i] == 'fc':
                in_weights = self.cnn_out_params
                out_weights =  self.cnn_out_params if (num_fc_done < cnn_num_fc_layers-1) else 2
                sigm = True if (num_fc_done < cnn_num_fc_layers-1) else False
                batchn = True if (num_fc_done < cnn_num_fc_layers-1) else False
                layersOut['layer'+str(i+1)] ,weights['w'+str(i+1)], biases['b'+str(i+1)] = self.fcLayer(prev_layer,
                                                    [in_weights , out_weights], sigm, batchn)
                num_fc_done += 1
                print("This is the FC NN Layer")
                print("This is the shape of the outputs of this layer: " + str(layersOut['layer'+str(i+1)].get_shape().as_list()))



            prev_layer = layersOut['layer'+str(i+1)]
            layer_counter += 1

            if layer_counter == cnn_num_layers:
                layersOut['output_values'] = layersOut['layer'+str(layer_counter)]
                # layersOut['output_values'] = tf.reshape(layersOut['output_values'], [out_weights, self.batch_size], name=None)
                layersOut['pred'] = tf.nn.softmax(layersOut['output_values'], dim=-1, name=None)
                # layersOut['pred'] = tf.transpose(layersOut['pred'], perm=None, name='transpose')
                break


        self.layersOut = layersOut
        self.weights = weights
        self.biases = biases

        return layersOut, weights


    def train(self, cnn_reg_on, cnn_reg_op):
        cost = self.cost_function( self.layersOut['output_values'], self.output, op='log-likelihood')
        cumCost = cost
        numEntries = len(self.weights)

        if cnn_reg_on:
            weightVals = self.weights.values()
            biasVals = self.biases.values()
            for i in range(numEntries):
                cumCost = self.add_regularization( cumCost, weightVals[i], self.w_lmbda[i], None, op=cnn_reg_op)
                cumCost = self.add_regularization( cumCost, biasVals[i], self.b_lmbda[i], None, op=cnn_reg_op)

        train_op = self.minimization_function(cumCost, self.learning_rate, self.beta1, self.beta2, self.op)
        return cumCost, train_op

class MultiModalNN(CNNLayers):
    def __init__(self,cnn_out, fcnet_out, output, batch_size=1,  learning_rate=1e-3, beta1=0.99, beta2=0.99, w_lmbda = None, b_lmbda = None, op='Rmsprop' ):
        CNNLayers.__init__(self)

        self.cnn_out = cnn_out
        self.fcnet_out = fcnet_out
        self.output = output
        self.w_lmbda = w_lmbda
        self.b_lmbda = b_lmbda
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.op = op

    def build_model(self, num_layers, hidden_units, sigmoid=True, batch_norm=True):

        cnn_shape = self.cnn_out.get_shape().as_list()
        fcnet_shape = self.fcnet_out.get_shape().as_list()

        newInShape = [cnn_shape[0], cnn_shape[1]+ fcnet_shape[1]]

        prev_shape = newInShape[1]
        concatIn = tf.concat(1, [self.cnn_out, self.fcnet_out])

        weights = {}
        layersOut = {}
        biases = {}

        prev_layer = concatIn

        for i in range(num_layers):
            lastSigm = True if i < (num_layers-1) else False
            lastBatch = True if i < (num_layers-1) else False
            layersOut['layer'+str(i+1)] ,weights['w'+str(i+1)] , biases['b'+str(i+1)]= self.fcLayer(prev_layer, [prev_shape, hidden_units[i]], lastSigm, lastBatch)
            prev_shape = hidden_units[i]
            prev_layer = layersOut['layer'+str(i+1)]

        layersOut['output_values'] = prev_layer
        layersOut['pred'] = tf.nn.softmax(layersOut['output_values'], dim=-1, name=None)
        # layersOut['pred'] = tf.reshape(layersOut['pred'], [1,self.batch_size], name=None)

        self.layersOut = layersOut
        self.weights = weights
        self.biases = biases

        return layersOut, weights

    def train(self):
        cost = self.cost_function(self.layersOut['pred'], self.output, op='log-likelihood')
        cumCost = cost
        numEntries = len(self.weights)

        weightVals = self.weights.values()
        biasVals = self.biases.values()
        for i in range(numEntries-2):
            cumCost = self.add_regularization( cumCost, weightVals[i], self.w_lmbda[i], None, op='l2')
            cumCost = self.add_regularization( cumCost, biasVals[i], self.b_lmbda[i], None, op='l2')

        train_op = self.minimization_function(cumCost, self.learning_rate, self.beta1, self.beta2, self.op)
        return cumCost, train_op


class ResidualNet(CNNLayers):
    def __init__(self, image, output, p_keep_conv, numResUnits, batch_size=1, learning_rate=1e-3, beta1=0.99, beta2=0.99, w_lmbda = None, b_lmbda = None, op='Rmsprop'):
        CNNLayers.__init__(self)
        self.input_image = image
        self.output = output
        self.dropout = p_keep_conv
        self.numResUnits = numResUnits

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.w_lmbda = w_lmbda
        self.b_lmbda = b_lmbda
        self.op = op

    def build_custom_units(self, resUnitNames, resUnitFilters, resUnitStrides):
        pass

    def build_model(self, in_conv_filter,in_conv_stride, in_pool, in_pool_stride, resUnit_filter, resUnit_filter_stride, resUnit_pool,
                        resUnit_pool_stride, resUnit, numFilter ):
        weights = {}
        layersOut = {}
        biases = {}

        layersOut['input'] = self.input_image
        layersOut['output'] = self.output
