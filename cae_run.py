import tensorflow as tf
import numpy as np
from utils import *
from cae_layers import *
from create_brain_binaries import _create_feature_binary
import os
import argparse
import json

# other constants
TRAIN = 0
TEST = 1
RUN_ALL = 2
CHECKPOINT_DIR = '/data/axel_ckpt/cae_working'
TRAIN_BINARY = '/data/train2.json'
TEST_BINARY = '/data/test2.json'

# originally in models.py for cae
class ConvAutoEncoder(CNNLayers):

    def __init__(self, train, data_list, input_dimensions, batch_size=1, learning_rate=1e-3, beta1=0.99, beta2=0.99, rho=0.4, lmbda = 0.6, op='Rmsprop'):
        CNNLayers.__init__(self)
        self.keys, self.input_image, self.input_data, self.output, self.p_keep = self.createVariables(train, data_list, batch_size, input_dimensions)
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

        prev_layer = self.input_image
        layer_shapes['w0'] =  self.input_image.get_shape().as_list()
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

        return self.keys, layer_outputs, weights, layer_shapes, encode, decode, self.input_image, self.input_data, self.output

    def train(self):
        cost = self.cost_function(self.decode, self.input_image, op='square')

        cumCost = cost
        numEntries = len(self.weights)

        weightVals = self.weights.values()
        #for i in range(numEntries):
        #    cumCost = self.add_regularization( cumCost, weightVals[i], self.lmbda[i], self.rho[i], op='kl')

        cumCost = self.add_regularization( cost, self.encode, self.lmbda, self.rho, op='kl')
        train_op = self.minimization_function(cumCost, self.learning_rate, self.beta1, self.beta2, self.op)
        return cumCost, train_op


def createAutoEncoderModel(train, data_list, batch_size):
    # hyperparameters
    filter_sz = [3,3]
    stride_sz = [1,3]
    learning_rate = 0.001
    beta1 = 0.99
    beta2 = None
    rho = 0.7
    lmbda = 0.6
    op = 'Rmsprop'
    batchOn = True
    input_dimensions = [91, 109, 91]

    allFilters = []
    for i in filter_sz:
        allFilters.append([i,i,i,1,1])
    allStrides = []
    for i in stride_sz:
        allStrides.append([1,i,i,i,1])

    numForwardLayers = len(allFilters)
    #lmbVec = [0.4]*numForwardLayers*2
    #rhoVec = [0.1]*numForwardLayers*2

    print("Creating the Convolutional AutoEncoder Object")
    cae = ConvAutoEncoder(train, data_list, input_dimensions, batch_size,
                    learning_rate, beta1, beta2, rho=rho, lmbda=lmbda, op=op)

    allNames = ["layer1_filters","layer2_filters","layer3_filters","layer4_filters","layer5_filters","layer6_filters",
                "layer7_filters","layer8_filters","layer9_filters","layer10_filters"]
    allRelu = [True]*numForwardLayers*2
    allBatch = [batchOn]*numForwardLayers*2

    # We do not need ReLUs in the encoder layer and the decode layer
    # DO NOT CHANGE UNLESS NECESSARY
    allRelu[numForwardLayers-1] = False
    allRelu[2*numForwardLayers-1] = False
    allBatch[numForwardLayers-1] = False
    allBatch[2*numForwardLayers-1] = False

    print("Building the Convolutional Autoencoder Model")
    keys, layer_outputs, weights, weight_shapes, encode, decode, brain_image, pheno_data, label = cae.build_model(numForwardLayers, allFilters, allStrides, allNames, allRelu, allBatch)

    print("Setting up the Training model of the Autoencoder")
    cost, train_op = cae.train()
    return keys, layer_outputs, weights, weight_shapes, encode, decode, brain_image, pheno_data, label, cost, train_op


def run_cae(state, input_dir, batch_size, max_steps):
    # binary_filelist: path of file containing binary file name list
    output_dir = input_dir + '_reduced'
    genDirectoryJSON(dataDir=input_dir)
    binary_filelist = SAMPLE_JSON

    keys, layer_outputs, weights, weight_shapes, encode, decode, brain_image, pheno_data, label, cost, train_op = createAutoEncoderModel(state==TRAIN, binary_filelist, batch_size)

    # layer_outputs, weights, cost, train_op = createCNNModel(train, binary_filelist)

    print("Created the entire model! YAY!")

    # Create a saver
    saver = tf.train.Saver(tf.all_variables())

    # Launch the graph in a session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
        init_op.run()

        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)
        ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        if state==TRAIN:
            # Get checkpoint at step: i_stopped
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print(ckpt.model_checkpoint_path)
                i_stopped = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            else:
                print('No checkpoint file found!')
                i_stopped = 0

        else: # testing (or running all files)
            # Get most recent checkpoint & start from beginning
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print(ckpt.model_checkpoint_path)
            i_stopped = 0


        for i in range(i_stopped, max_steps):
            print("Running iteration {} of TF Session".format(i))
            if state==TRAIN:
                _, loss = sess.run([train_op, cost])
            else:
                loss = sess.run(cost)
            # print("The current loss is: " + str(loss))


            # Checkpoint model at each 10 iterations (only during training)
            if state==TRAIN and (i != 0 and (i % 20 == 0 or (i+1) == max_steps)):
                checkpoint_path = os.path.join(CHECKPOINT_DIR, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=i)

            # If running all files
            if state==RUN_ALL:
                # current_file = filenames[i].split(".")[0].split("/")[-1]
                # Saving output of CAE to binary files
                encoded_image = np.asarray(sess.run(encode))
                # Get filename
                k = sess.run(keys)
                filename = k[0][:-2]
                print "Current file read: " + os.path.basename(filename)
                # Get label and phenotype data
                patient_label = np.asarray(sess.run(label))
                patient_pheno = np.asarray(sess.run(pheno_data))
                # ourput image currently: 31x37x31

                out_filename = os.path.join(output_dir, os.path.basename(filename).replace('.','_reduced.'))
                print out_filename
                _create_feature_binary(patient_pheno, encoded_image, patient_label, out_filename)

        if state==TRAIN:
            coord.request_stop()
            coord.join(stop_grace_period_secs=10)

        # Visualization of CAE output
        encodeLayer = np.asarray(sess.run(encode))
        decodeLayer = np.asarray(sess.run(decode))
        inputImage = np.asarray(sess.run(brain_image))
        print np.shape(inputImage)

        mat2visual(encodeLayer[0, 0,:,:,:, 0], [10, 15, 19], 'encodedImage.png', 'auto')
        mat2visual(decodeLayer[0, 0,:,:,:, 0], [40, 55, 60], 'decodedImage.png', 'auto')
        mat2visual(inputImage[0, :,:,:, 0], [40, 55, 60], 'inputImage.png', 'auto')

def applyCAE(state=RUN_ALL, input_dir=''):
    if state==TRAIN:  # We have 963 train patients
        binary_filelist = TRAIN_BINARY
        output_dir = None
        batch_size = 32
        max_steps = 5000         # can be changed
    elif state==TEST: # We have 108 train patients
        binary_filelist = TEST_BINARY
        output_dir = None
        max_steps = 108
    elif state==RUN_ALL:
        outputFolder = input_dir + '_reduced'
        if outputFolder != None:
            if not os.path.exists(outputFolder):
                os.makedirs(outputFolder)

        batch_size = 1
        max_steps = len(os.listdir(input_dir))

    if os.path.exists(SAMPLE_DIR):
        shutil.rmtree(SAMPLE_DIR)
    os.mkdir(SAMPLE_DIR)

    run_cae(state, input_dir, batch_size, max_steps)

if __name__ == "__main__":
    applyCAE(state=RUN_ALL, input_dir='/data/augmented_swap_partial_steal_28')
