import tensorflow as tf
import numpy as np
from utils import *
from layers import *
from models import *
from input_brain import *
import os
import create_brain_binaries
import argparse
from constants import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/data/train',
                           """Directory where to write event logs """)

tf.app.flags.DEFINE_string('train_binaries', '/data/train.json',
                           """File containing list of binary filenames used for training """)
tf.app.flags.DEFINE_string('test_binaries', '/data/test.json',
                           """File containing list of binary filenames used for testing """)
tf.app.flags.DEFINE_string('all_binaries', '/data/all.json',
                           """File containing list of all the binary filenames """)

# Auto Encoder compressed files
tf.app.flags.DEFINE_string('reduced_dir', '/data/binaries_reduced',
                           """File containing list of all the binary filenames """)
tf.app.flags.DEFINE_string('reduced_train_binaries', '/data/reduced_train.json',
                           """File containing list of binary filenames used for training """)
tf.app.flags.DEFINE_string('reduced_test_binaries', '/data/reduced_test.json',
                           """File containing list of binary filenames used for testing """)
tf.app.flags.DEFINE_string('reduced_all_binaries', '/data/reduced_all.json',
                           """File containing list of all the binary filenames """)

# Run Model flags
tf.app.flags.DEFINE_boolean('train', True, """Training the model """)
tf.app.flags.DEFINE_boolean('test', True, """Testing the model """)
tf.app.flags.DEFINE_boolean('cae', True, """Run the Convolutional AutoEncoder """)


def createAutoEncoderModel(train, data_list, input_dimensions, batch_size):

    # hyperparameters
    filter_sz = CAE_FILTER_SZ
    stride_sz = CAE_STRIDE_SZ
    num_filters = CAE_NUM_FILTERS
    learning_rate = CAE_LEARNING_RATE
    beta1 = CAE_BETA_1
    beta2 = CAE_BETA_2 
    rho = CAE_RHO
    lmbda = CAE_LAMBDA
    op = CAE_OP
    batchOn = CAE_BATCH_NORM

    prev_filter = 1
    cnt = 0
    allFilters = []
    for i in filter_sz:
        allFilters.append([i,i,i,prev_filter,num_filters[cnt]])
        prev_filter = num_filters[cnt]
        cnt += 1

    allStrides = []
    for i in stride_sz:
        allStrides.append([1,i,i,i,1])

    numForwardLayers = CAE_NUM_ENCODE_LAYERS
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
    layer_outputs, weights, weight_shapes, encode, \
    decode, brain_image, pheno_data, label = cae.build_model(numForwardLayers,
                            allFilters, allStrides, allNames, allRelu, allBatch)


    print("Setting up the Training model of the Autoencoder")
    cost, train_op = cae.train()
    return layer_outputs, weights, weight_shapes, encode, decode, \
                        brain_image, pheno_data, label, cost, train_op


def createCNNModel(train, data_list, input_dimensions, batch_size, multiModal=False):

    numLayers = 8
    regConstants = [0.6]*numLayers
    print("Creating the Convolutional Neural Network Object")

    deepCnn = ConvNN(train, data_list, input_dimensions, numLayers, batch_size,
                        CNN_LEARNING_RATE, CNN_BETA_1, CNN_BETA_2, lmbda = CNN_REG_CONSTANTS , op=CNN_OP)
    print("Building the Deep CNN Model")
    # layersOut, weights, image, data, label = deepCnn.build_model(True, False)

    # layersOut, weights =  deepCnn.build_model(True, False)
    layersOut, weights =  deepCnn.build_model(CONV_ARCH, CNN_NUM_LAYERS, CNN_NUM_FC_LAYERS, 
                        CNN_FILTER_SZ, CNN_NUM_FILTERS, CNN_STRIDE_SZ, CNN_POOL_SZ, CNN_POOL_STRIDE_SZ, CNN_BATCH_NORM)

    if multiModal:
        # return layersOut, weights, image, data, label
        return layersOut, weights

    print("Setting up the Training model of the Deep CNN")
    cost, train_op = deepCnn.train()

    return layersOut, weights, cost, train_op


def createVanillaNN(train, data_list, input_dimensions, batch_size, multiModal=False, image=None, data=None, label=None):

    learning_rate = NN_LEARNING_RATE
    beta1 = NN_BETA_1
    beta2 = NN_BETA_2
    op = NN_OP
    batchOn = NN_BATCH_NORM
    sigmoidOn = NN_SIGMOID

    regConstants = NN_REG_CONSTANTS
    hidden_units = NN_HIDDEN_UNITS

    print("Creating the Vannil Neural Network Object")

    deepNN = NeuralNetwork(train, data_list, input_dimensions, batch_size,
                            learning_rate, beta1, beta2, lmbda=regConstants, op=op)
                            # image=image, data=data, label=label)


    print("Building the Vanilla Neural Network Model")
    layersOut, weights = deepNN.build_model(len(NN_HIDDEN_UNITS), hidden_units, sigmoidOn, batchOn, NN_MMLAYER)

    if multiModal:
        return layersOut, weights

    print("Setting up the Training model of the Vanilla Neural Network ")
    cost, train_op = deepNN.train()

    return layersOut, weights, cost, train_op


def createMultiModalNN(train, binary_filelist, input_dimensions, batch_size):

    # layersCnn, weightsCnn, image, data, label = createCNNModel(train, binary_filelist, input_dimensions, batch_size, True)
    layersCnn, weightsCnnl = createCNNModel(train, binary_filelist, input_dimensions, batch_size, True)

    # layersFc, weightsFc = createVanillaNN(train, binary_filelist, input_dimensions, batch_size, True, image, data, label)
    layersFc, weightsFc = createVanillaNN(train, binary_filelist, input_dimensions, batch_size, True)

    learning_rate = MMNN_LEARNING_RATE 
    beta1 = MMNN_BETA_1 
    beta2 = MMNN_BETA_2
    op = MMNN_OP
    batchOn = MMNN_BATCH_NORM

    numLayers = len(MMNN_HIDDEN_UNITS)
    regConstants = MMNN_REG_CONSTANTS
    hidden_units = MMNN_HIDDEN_UNITS

    print("Creating the Multi Modal Convolutional Neural Network Object")

    deepMultiNN = MultiModalNN(train, binary_filelist, layersCnn['layer'+str(CNN_MMLAYER)+'-fc'],layersFc['layer'+str(NN_MMLAYER)], layersCnn['output'], batch_size,
                learning_rate, beta1, beta2, lmbda = regConstants, op='Rmsprop')


    print("Building the Multi Modal NN Model")
    layersOut, weights = deepMultiNN.build_model(numLayers, hidden_units, True, False)

    print("Setting up the Training model of the Multi Modal NN Model")
    cost, train_op = deepMultiNN.train()

    return layersOut, weights, cost, train_op



def run_model(train, model, binary_filelist, run_all, batch_size, max_steps):
    input_dimensions = [31, 37, 31]
    if model == 'cae':
        input_dimensions = [91, 109, 91]
        layer_outputs, weights, weight_shapes, encode, decode, brain_image, \
        pheno_data, label, cost, train_op = createAutoEncoderModel(train,
                                binary_filelist, input_dimensions, batch_size)
    elif model == 'cnn':
        layer_outputs, weights, cost, train_op = createCNNModel(train,
                                binary_filelist, input_dimensions, batch_size)
    elif model == 'nn':
        layer_outputs, weights, cost, train_op = createVanillaNN(train,
                                binary_filelist, input_dimensions, batch_size)
    elif model == 'mmnn':
        layer_outputs, weights, cost, train_op = createMultiModalNN(train,
                                binary_filelist, input_dimensions, batch_size)
    else:
        print("Kindly put in the correct model")

    print("Reading the binary file list: " + binary_filelist)
    print("Using the following input dimensions: " + str(input_dimensions))

    print("Created the entire model! YAY!")
    # Create a saver
    saver = tf.train.Saver(tf.all_variables())

    # Launch the graph in a session
    with tf.Session() as sess:

        init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())

        init_op.run()

        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if train:
            # Get checkpoint at step: i_stopped
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Fetching checkpoint data from:")
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


        compressed_filelist = []
        for i in range(i_stopped, max_steps):
            print("Running iteration {} of TF Session".format(i))
            if train:
                _, loss = sess.run([train_op, cost])
            else:
                loss = sess.run(cost)
            print("The current loss is: " + str(loss))

            # Checkpoint model at each 100 iterations
            should_save = i != 0 and i % 1000 == 0 or (i+1) == max_steps
            if should_save and train:
                checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=i)

            # If running all files for CAE
            if not train and run_all and model == 'cae':
                # Saving output of CAE to binary files
                encoded_image = np.asarray(sess.run(encode))
                # Get label and phenotype data
                patient_label = np.asarray(sess.run(label))
                patient_pheno = np.asarray(sess.run(pheno_data))
                # ourput image currently: 31x37x31

                bin_path = create_brain_binaries.create_compressed_binary(
                                        patient_pheno, encoded_image,
                                        patient_label, FLAGS.reduced_dir, str(i+1))
                compressed_filelist.append(bin_path)

        coord.request_stop()
        coord.join(stop_grace_period_secs=10)


        # Visualization of CAE output
        if model == 'cae':
            # If running all files for CAE
            if not train and run_all:
                create_brain_binaries.save_and_split(compressed_filelist,
                                                     output_binary_filelist,
                                                     FLAGS.reduced_train_binaries,
                                                     FLAGS.reduced_test_binaries)

            encodeLayer = np.asarray(sess.run(encode))
            decodeLayer = np.asarray(sess.run(decode))
            inputImage = np.asarray(sess.run(brain_image))

            mat2visual(encodeLayer[0, 0,:,:,:, 0], [10, 15, 19], 'encodedImage.png', 'auto')
            mat2visual(decodeLayer[0, 0,:,:,:, 0], [40, 55, 60], 'decodedImage.png', 'auto')
            mat2visual(inputImage[0, :,:,:, 0], [40, 55, 60], 'inputImage.png', 'auto')



def main(_):
    # Argument parsing
    parser = argparse.ArgumentParser(description='Evaluation procedure for Salami CNN.')
    network_group = parser.add_mutually_exclusive_group()
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument('--train', action="store_true", help='Training the model')
    data_group.add_argument('--test', action="store_true", help='Testing the model')
    network_group.add_argument('--model', choices=['cae', 'cnn', 'nn', 'mmnn'],
                        default='mmnn', help='Select model to run.')
    parser.add_argument('--chkPointDir', dest='chkPt', default='/data/ckpt',
                        help='Directory to save the checkpoints. Default is /data/ckpt')
    parser.add_argument('--numIters', dest='numIters', default=200, type=int,
                        help='Number of Training Iterations. Default is 200')
    args = parser.parse_args()

    binary_filelist = None
    batch_size = 1
    max_steps = 1071
    run_all = False

    if args.train:  # We have 963 train patients
        if args.model == 'cae':
            binary_filelist = FLAGS.train_binaries
        else:
            binary_filelist = FLAGS.reduced_train_binaries
        # batch_size = 32
        max_steps = args.numIters
    elif args.test: # We have 108 train patients
        if args.model == 'cae':
            binary_filelist = FLAGS.test_binaries
        else:
            binary_filelist = FLAGS.reduced_test_binaries
        max_steps = 108
    else:
        if args.model == 'cae':
            binary_filelist = FLAGS.all_binaries
        else:
            binary_filelist = FLAGS.reduced_all_binaries
        run_all = True

    # set the checkpoint directory.
    if not os.path.exists(args.chkPt):
        print "Directory '%s' does not exist." % args.chkPt
    FLAGS.checkpoint_dir = args.chkPt

    run_model(args.train, args.model, binary_filelist, run_all, batch_size, max_steps)


if __name__ == "__main__":
    tf.app.run()
