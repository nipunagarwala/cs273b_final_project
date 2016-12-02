import tensorflow as tf
import numpy as np
from utils import *
from layers import *
from models import *
from input_brain import *
import os
from constants import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/data/train',
                           """Directory where to write event logs """)

tf.app.flags.DEFINE_string('ae_train_binaries', '/data/reduced_train.json',
                           """File containing list of binary filenames used for training """)
tf.app.flags.DEFINE_string('ae_test_binaries', '/data/reduced_test.json',
                           """File containing list of binary filenames used for testing """)
tf.app.flags.DEFINE_string('ae_all_binaries', '/data/reduced_all.json',
                           """File containing list of all the binary filenames """)

tf.app.flags.DEFINE_string('train_binaries', '/data/train.json',
                           """File containing list of binary filenames used for training """)
tf.app.flags.DEFINE_string('test_binaries', '/data/test.json',
                           """File containing list of binary filenames used for testing """)
tf.app.flags.DEFINE_string('all_binaries', '/data/all.json',
                           """File containing list of all the binary filenames """)

# Convolutional Auto Encoder compressed files
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
tf.app.flags.DEFINE_boolean('ae', True, """Run the AutoEncoder """)


def createVariables(train, binary_filelist, input_dimensions, batch_size):
    # train: Boolean
    # data_list: Path of a file containing a list of all binary data file paths
    # batch_size: int
    p_keep_conv = tf.placeholder("float")
    X_image, X_data, Y = inputs(train, binary_filelist, batch_size, input_dimensions)
    return X_image, X_data, Y, p_keep_conv


def createAutoEncoderModel(data, output, p_keep_conv, batch_size):

    # hyperparameters
    learning_rate = AE_LEARNING_RATE
    beta1 = AE_BETA_1
    beta2 = AE_BETA_2
    rho = AE_RHO
    lmbda = AE_LAMBDA
    op = AE_OP
    batchOn = AE_BATCH_NORM

    hidden_units = AE_HIDDEN_UNITS

    numForwardLayers = len(hidden_units)
    #lmbVec = [0.4]*numForwardLayers*2
    #rhoVec = [0.1]*numForwardLayers*2

    print("Creating the AutoEncoder Object")
    ae = AutoEncoder(data, output, p_keep_conv, batch_size,
                     learning_rate, beta1, beta2, rho=rho, lmbda=lmbda, op=op)

    allRelu = [True]*numForwardLayers*2
    allBatch = [batchOn]*numForwardLayers*2

    # We do not need ReLUs in the encoder layer and the decode layer
    # DO NOT CHANGE UNLESS NECESSARY
    allRelu[numForwardLayers-1] = False
    allRelu[2*numForwardLayers-1] = False
    allBatch[numForwardLayers-1] = False
    allBatch[2*numForwardLayers-1] = False

    print("Building the Autoencoder Model")
    layer_outputs, weights, weight_shapes, encode, decode, pheno_data, label \
                = ae.build_model(hidden_units, allRelu, allBatch)

    print("Setting up the Training model of the Autoencoder")
    cost, train_op = ae.train()

    return layer_outputs, weights, encode, decode, \
                        pheno_data, label, cost, train_op


def createConvAutoEncoderModel(image, output, p_keep_conv,batch_size):

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
    cae = ConvAutoEncoder(image, output, p_keep_conv,batch_size,
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
    layer_outputs, weights, encode, \
    decode, brain_image = cae.build_model(numForwardLayers,
                            allFilters, allStrides, allNames, allRelu, allBatch)


    print("Setting up the Training model of the Autoencoder")
    cost, train_op = cae.train()
    return layer_outputs, weights, encode, decode, \
                        brain_image, cost, train_op


def createCNNModel(image, output, p_keep_conv,batch_size, multiModal=False):

    numLayers = 8
    regConstants = [0.6]*numLayers
    print("Creating the Convolutional Neural Network Object")

    deepCnn = ConvNN(image, output, p_keep_conv, numLayers, batch_size,
                        CNN_LEARNING_RATE, CNN_BETA_1, CNN_BETA_2, w_lmbda = CNN_REG_CONSTANTS_WEIGHTS,
                            b_lmbda = CNN_REG_CONSTANTS_BIAS , op=CNN_OP)
    print("Building the Deep CNN Model")
    # layersOut, weights, image, data, label = deepCnn.build_model(True, False)

    # layersOut, weights =  deepCnn.build_model(True, False)
    layersOut, weights =  deepCnn.build_model(CONV_ARCH, CNN_NUM_LAYERS, CNN_NUM_FC_LAYERS,
                        CNN_FILTER_SZ, CNN_NUM_FILTERS, CNN_STRIDE_SZ, CNN_POOL_SZ, CNN_POOL_STRIDE_SZ, CNN_BATCH_NORM)

    if multiModal:
        # return layersOut, weights, image, data, label
        return layersOut, weights

    print("Setting up the Training model of the Deep CNN")
    cost, train_op = deepCnn.train( CNN_REG_ON, CNN_REG_OP)

    return layersOut, weights, cost, train_op


def createVanillaNN(data, output, p_keep_conv,batch_size, multiModal=False):

    learning_rate = NN_LEARNING_RATE
    beta1 = NN_BETA_1
    beta2 = NN_BETA_2
    op = NN_OP
    batchOn = NN_BATCH_NORM
    sigmoidOn = NN_SIGMOID

    regConstants = NN_REG_CONSTANTS_WEIGHTS
    hidden_units = NN_HIDDEN_UNITS

    print("Creating the Vannil Neural Network Object")

    deepNN = NeuralNetwork(data, output, p_keep_conv, batch_size,
                            learning_rate, beta1, beta2, w_lmbda=regConstants, b_lmbda = NN_REG_CONSTANTS_BIAS, op=op)
                            # image=image, data=data, label=label)


    print("Building the Vanilla Neural Network Model")
    layersOut, weights = deepNN.build_model(len(NN_HIDDEN_UNITS), hidden_units, sigmoidOn,
                                    batchOn, NN_MMLAYER)

    if multiModal:
        return layersOut, weights

    print("Setting up the Training model of the Vanilla Neural Network ")
    cost, train_op = deepNN.train(NN_REG_ON , NN_REG_OP)

    return layersOut, weights, cost, train_op


def createMultiModalNN(image, data, output, p_keep_conv, batch_size):

    # layersCnn, weightsCnn, image, data, label = createCNNModel(train, binary_filelist, input_dimensions, batch_size, True)
    layersCnn, weightsCnnl = createCNNModel(image, output, p_keep_conv,batch_size, True)

    # layersFc, weightsFc = createVanillaNN(train, binary_filelist, input_dimensions, batch_size, True, image, data, label)
    layersFc, weightsFc = createVanillaNN(data, output, p_keep_conv,batch_size,  True)

    learning_rate = MMNN_LEARNING_RATE
    beta1 = MMNN_BETA_1
    beta2 = MMNN_BETA_2
    op = MMNN_OP
    batchOn = MMNN_BATCH_NORM

    numLayers = len(MMNN_HIDDEN_UNITS)
    regConstants = MMNN_REG_CONSTANTS_WEIGHTS
    hidden_units = MMNN_HIDDEN_UNITS

    print("Creating the Multi Modal Convolutional Neural Network Object")

    deepMultiNN = MultiModalNN( layersCnn['layer'+str(CNN_MMLAYER)+'-fc'],layersFc['layer'+str(NN_MMLAYER)], layersCnn['output'], batch_size,
                learning_rate, beta1, beta2, w_lmbda = regConstants, b_lmbda = MMNN_REG_CONSTANTS_BIAS, op=MMNN_OP)


    print("Building the Multi Modal NN Model")
    layersOut, weights = deepMultiNN.build_model(numLayers, hidden_units, True, MMNN_BATCH_NORM)

    print("Setting up the Training model of the Multi Modal NN Model")
    cost, train_op = deepMultiNN.train()

    return layersOut, weights, cost, train_op



def run_model(train, model, binary_filelist, run_all, batch_size, max_steps, overrideChkpt):
    if model=='cae':
        input_dimensions = [91, 109, 91]
    else:
        input_dimensions = [31, 37, 31]

    image, data, output, p_keep_conv = createVariables(train, binary_filelist, input_dimensions, batch_size)

    if model == 'ae':
        layer_outputs, weights, encode, decode, \
        pheno_data, label, cost, train_op = createAutoEncoderModel(data, output, p_keep_conv, batch_size)
    elif model == 'cae':
        layer_outputs, weights, encode, decode, brain_image, \
        cost, train_op = createConvAutoEncoderModel(image, output, p_keep_conv, batch_size)
    elif model == 'cnn':
        layer_outputs, weights, cost, train_op = createCNNModel(image, output, p_keep_conv, batch_size)
    elif model == 'nn':
        layer_outputs, weights, cost, train_op = createVanillaNN(data, output, p_keep_conv, batch_size)
    elif model == 'mmnn':
        layer_outputs, weights, cost, train_op = createMultiModalNN(image, data, output, p_keep_conv, batch_size)
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

        i_stopped = setup_checkpoint(train, sess, saver, FLAGS.checkpoint_dir)

        compressed_filelist = []
        for i in range(i_stopped, max_steps):
            print("Running iteration {} of TF Session".format(i))
            if train:
                _, loss = sess.run([train_op, cost])
            else:
                loss = sess.run(cost)
            print("The current loss is: " + str(loss))
            # print("Current predicted labels are: " + str(layer_outputs['pred'].eval()))

            # Checkpoint model at each 100 iterations
            should_save = i != 0 and i % 1000 == 0 or (i+1) == max_steps
            if should_save and train:
                checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=i)

            # If running all files for CAE
            if not train and run_all and model == 'cae':
                bin_path = create_CEA_reduced_binary(sess, encode, output,
                                                    data, FLAGS, i)
                compressed_filelist.append(bin_path)

        coord.request_stop()
        coord.join(stop_grace_period_secs=10)


        # CAE Output
        generate_CAE_output(train, model, run_all, sess, encode, decode, brain_image,
                            compressed_filelist, output_binary_filelist, FLAGS)



def main(_):
    # Argument parsing
    args = extract_parser()

    # Create conditional variables
    binary_filelist, batch_size, max_steps, run_all = create_conditions(args, FLAGS)

    # Set the checkpoint directory.
    if not os.path.exists(args.chkPt):
        print "Directory '%s' does not exist." % args.chkPt
    FLAGS.checkpoint_dir = args.chkPt

    run_model(args.train, args.model, binary_filelist, run_all, batch_size, max_steps, args.overrideChkpt)


if __name__ == "__main__":
    tf.app.run()
