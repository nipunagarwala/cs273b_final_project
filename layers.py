import tensorflow as tf
import numpy as np
from input_brain import *
from utils import *
# from utils_visual import *


# Define custom API for creating and adding layers to NN Model
# Wrapper around Tensorflow API, for ease of use and readibility

class Layers(object):

    def __init__(self):
        pass

    ''' Initializes the weights based on the std dev set in the constructor

    '''
    def get_fans(self, shape):
        if len(shape) == 1:
            fan_in = 20000
            fan_out = fan_in
        elif len(shape) == 2:
            fan_in = shape[0]
            fan_out = shape[1]
        else:
            receptive_field_size = shape[0] * shape[1] * shape[2]
            fan_in = shape[3] * receptive_field_size
            fan_out = shape[4] * receptive_field_size
        return fan_in, fan_out

    def init_weights(self, shape):
        fan_in, fan_out = self.get_fans(shape)
        stddev = tf.sqrt(2.0 / fan_in)
        return tf.Variable(tf.random_normal(shape, stddev=stddev))


    def dropout(self, prev_layer,  p_keep):
        next_layer = tf.nn.dropout(prev_layer, p_keep)
        return next_layer

    def sigmoid(self, prev_layer):
        next_layer = tf.sigmoid(prev_layer)
        return next_layer

    def tanh(self, prev_layer):
        next_layer = tf.tanh(prev_layer)
        return next_layer


    def batch_norm(self, prev_layer, axes, beta_shape, scale_shape, phase_train, var_eps = 1e-6):
        def mean_var_with_update():
            ema_apply_op = ema.apply([mu, sigma])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(mu), tf.identity(sigma)

        mu, sigma = tf.nn.moments(prev_layer, axes)
        beta = self.init_weights(beta_shape)
        scale = self.init_weights(scale_shape)
        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(mu), ema.average(sigma)))

        next_layer = tf.nn.batch_normalization(prev_layer, mean, var, beta, scale, var_eps) #1e-3)
        # next_layer = tf.nn.batch_normalization(prev_layer, mu, sigma, beta, scale, var_eps)
        return next_layer

    def fcLayer(self, prev_layer, wshape, phase_train, sigmoid=True, batch_norm=False):
        wOut = self.init_weights(wshape)
        b = self.init_weights([wshape[1]])
        next_layer = tf.add(tf.matmul(prev_layer, wOut), b)
        if batch_norm:
            next_layer = self.batch_norm(next_layer, [0], [wshape[1]], [wshape[1]], phase_train)
        if sigmoid:
            next_layer = self.relu(next_layer)


        return next_layer, wOut, b

    def cost_function(self, model_output, Y, op='square'):
        cost = None
        if  op == 'square':
            cost = tf.reduce_mean(tf.square(tf.sub(model_output,Y)))
        elif op == 'sigmoid':
            epsilon = 10e-5
            output = tf.clip_by_value(model_output, epsilon, 1 - epsilon)
            # Create logit of output
            output_logit = tf.log(output / (1 - output))
            # Reshape for comparison
            cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(output_logit, Y))
        elif op == 'softmax':
            Y_control = tf.ones_like(Y) - Y
            Y_full = tf.concat(1, [Y_control, Y])
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model_output , Y_full))
        elif op == 'log-likelihood':
            Y_control = tf.ones_like(Y) - Y
            Y_full = tf.concat(1, [Y_control, Y])
            epsilon = 10e-6
            prob = tf.nn.softmax(model_output, dim=-1, name=None)
            actionLabels = tf.mul(prob, Y_full)
            sumRes = tf.reduce_sum(actionLabels, 1)
            actionLikelihood = tf.clip_by_value( tf.log(sumRes), -1e4, 1000)
            actionLikelihood = tf.reduce_mean(actionLikelihood)
            cost = -actionLikelihood

        return cost

    def minimization_function(self, cost, learning_rate, beta1, beta2, opt='Rmsprop'):
        train_op = None
        if opt == 'Rmsprop':
            if not beta2:
                beta2 = 0
            train_op = tf.train.RMSPropOptimizer(learning_rate, beta1, beta2).minimize(cost)
        elif opt == 'adam':
            if not beta2:
                beta2 = 0.99
            train_op = tf.train.AdamOptimizer(learning_rate, beta1, beta2).minimize(cost)
        elif opt == 'adagrad':
            train_op = tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value=0.1).minimize(cost)

        return train_op

    def add_regularization(self, loss, wgt, lmbda, rho, op='kl'):
        nextLoss = None
        if op == 'l2':
            nextLoss = tf.add(loss, tf.mul(lmbda,tf.reduce_mean(tf.square(wgt))))
        elif op == 'kl':
            nextLoss = tf.add(loss, tf.mul(lmbda, self.kl_sparse_regularization(wgt, lmbda, rho)))
        elif op == 'l1':
            nextLoss = tf.add(loss, tf.mul(lmbda,tf.reduce_mean(tf.abs(wgt))))
        return nextLoss

    def kl_sparse_regularization(self, wgt, lmbda, rho):
        rho_hat = tf.reduce_mean(wgt)
        invrho = tf.sub(tf.constant(1.), rho)
        invrhohat = tf.sub(tf.constant(1.), rho_hat)
        logrho = tf.add(tf.abs(self.logfunc(rho,rho_hat)), tf.abs(self.logfunc(invrho, invrhohat)))
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

    def conv_layer(self, prev_layer_out, w_shape, layer_stride, w_name, phase_train, num_dim = '2d', padding='SAME',if_relu = True, batchNorm = True):
        w_conv = self.init_weights(w_shape)

        numFilters = w_shape[len(w_shape)-1]
        b = self.init_weights([numFilters])

        nextLayer = None
        if num_dim == '3d':
            nextLayer = tf.add(tf.nn.conv3d(prev_layer_out, w_conv,
                            strides=layer_stride, padding=padding,name=w_name),b)
        else:
            nextLayer = tf.add(tf.nn.conv2d(prev_layer_out, w_conv,
                            strides=layer_stride, padding=padding,name=w_name),b)

        if batchNorm:
            nextLayer = self.batch_norm(nextLayer, [0,1,2,3], [numFilters], [numFilters], phase_train)

        if if_relu:
            nextLayer = self.relu(nextLayer)


        return nextLayer, w_conv, b


    def deconv_layer(self, prev_layer_out, filter_shape, out_shape, layer_stride, w_name, phase_train, num_dim = '2d',padding='SAME', if_relu = True, batchNorm = True):
        w_deconv = self.init_weights(filter_shape)

        numFilters =filter_shape[len(filter_shape)-2]
        b = self.init_weights([numFilters])

        nextLayer = None

        if num_dim == '3d':
            nextLayer = tf.add(tf.nn.conv3d_transpose(prev_layer_out, w_deconv, out_shape,
                            strides=layer_stride, padding=padding),b)
        else:
            nextLayer = tf.add(tf.nn.conv2d_transpose(prev_layer_out, w_deconv, out_shape,
                            strides=layer_stride, padding=padding),b)

        if batchNorm:
            nextLayer = self.batch_norm(nextLayer, [0,1,2,3], [numFilters], [numFilters], phase_train)

        if if_relu:
            nextLayer = self.relu(nextLayer)


        return nextLayer, w_deconv, b

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


class ResidualUnit(CNNLayers):

    def __init__(self, name, num_in_filters, num_mid_filter, num_end_filters, num_in_str, num_mid_str, num_end_str, filter_size):
        self.name = name
        self.num_in_filters = num_in_filters
        self.num_mid_filter = num_mid_filter
        self.num_end_filters = num_end_filters
        self.num_in_str = num_in_str
        self.num_mid_str = num_mid_str
        self.num_end_str = num_end_str
        self.filter_size = filter_size


    def build_residual_unit(self, prev_layer_out, layer_count, weight_dict, bias_dict, num_dim = '2d', padding='SAME'):
        input_layer = prev_layer_out
        prev_num_filters = prev_layer_out.get_shape().as_list()[4]

        filterIncSz = [1,1,1,prev_num_filters, self.num_in_filters]
        layInStride = [1,self.num_in_str,self.num_in_str,self.num_in_str,1]
        wInName = 'layer'+str(layer_count+1)
        nextLayer, weight_dict[wInName], bias_dict[wInName] = self.conv_layer(input_layer, filterIncSz, layInStride, wInName, num_dim,
                                                            padding,if_relu = True, batchNorm = True)


        filterMidSz = [self.filter_size,self.filter_size,self.filter_size,self.num_in_filters, self.num_mid_filters]
        layMidstride = [1,self.num_mid_str,self.num_mid_str,self.num_mid_str,1]
        wMidName = 'layer'+str(layer_count+2)
        nextLayer, weight_dict[wMidName ], bias_dict[wMidName ] = self.conv_layer(nextLayer, filterMidSz, layMidstride, wMidName, num_dim,
                                                            padding,if_relu = True, batchNorm = True)


        filterEndSz = [1,1,1,self.num_mid_filters, self.num_end_filters]
        layEndstride = [1,self.num_end_str,self.num_end_str,self.num_end_str,1]
        wEndName = 'layer'+str(layer_count+3)
        lastLayer, weight_dict[wEndName], bias_dict[wEndName] = self.conv_layer(nextLayer, filterEndSz, layEndstride, wEndName, num_dim,
                                                            padding,if_relu = False, batchNorm = True)


        if self.num_in_str > 1:
            reshapeFilterSz = [1,1,1,prev_num_filters, self.num_end_filters]
            wReshapeName = 'in'+str(layer_count+3)
            input_layer, _, _ = self.conv_layer(input_layer, reshapeFilterSz, layInStride, wReshapeName,
                                                num_dim, padding, if_relu = False, batchNorm = batch_norm)

        output_layer = input_layer + lastLayer
        output_layer = self.relu(output_layer)

        return output_layer, layer_count+3
