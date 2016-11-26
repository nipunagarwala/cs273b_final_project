import tensorflow as tf
import numpy as np
from utils import *
from layers import *
from models import *
from input_brain import *
import os

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/data/train',
                           """Directory where to write event logs """)
tf.app.flags.DEFINE_string('checkpoint_dir', '/data/ckpt',
                           """Directory where to write checkpoints """)
tf.app.flags.DEFINE_integer('max_steps', 100,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Batch size being fed in.""")
tf.app.flags.DEFINE_string('train_binaries', '/data/train.json',
                           """File containing list of binary filenames used for training """)
tf.app.flags.DEFINE_boolean('train', True,
                           """Boolean describing training mode """)


def createAutoEncoderModel(train, data_list):

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
    cae = ConvAutoEncoder(train, data_list, [FLAGS.batch_size, 45, 54, 45, 1], [FLAGS.batch_size, 45, 54, 45, 1], 
                          FLAGS.batch_size, learning_rate, beta1, beta2, rho=rho, lmbda=lmbda, op=op)
    
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
    layer_outputs, weights, weight_shapes, encode, decode = cae.build_model(numForwardLayers, allFilters, allStrides, allNames, allRelu, allBatch)

    print("Setting up the Training model of the Autoencoder")
    cost, train_op = cae.train()
    return layer_outputs, weights, weight_shapes, encode, decode, cost, train_op





def createCNNModel(train, data_list):

    numLayers = 7
    regConstants = [0.6]*numLayers
    print("Creating the Convolutional Neural Network Object")
    deepCnn = ConvNN(train, data_list, [FLAGS.batch_size, 45, 54, 45, 1], [FLAGS.batch_size, 45, 54, 45, 1], numLayers , FLAGS.batch_size, 0.001, 0.99, None, lmbda = regConstants, op='Rmsprop')

    print("Building the Deep CNN Model")
    layersOut, weights = deepCnn.build_model()

    print("Setting up the Training model of the Deep CNN")
    cost, train_op = deepCnn.train()

    return layersOut, weights, cost, train_op







def main():
    layer_outputs, weights, weight_shapes, encode, decode, cost, train_op = createAutoEncoderModel(True, FLAGS.train_binaries)

    # layer_outputs, weights, cost, train_op = createCNNModel()

    # X, Y, encode, decode, cost, train_op = createModel()
    print("Created the entire model! YAY!")

    # Create a saver
    saver = tf.train.Saver(tf.all_variables())

    # Launch the graph in a session
    with tf.Session() as sess:

        init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())

        init_op.run()

        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        # # Get checkpoint at step: i_stopped
        # ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        # if ckpt and ckpt.model_checkpoint_path:
        #     saver.restore(sess, ckpt.model_checkpoint_path)
        #     print(ckpt.model_checkpoint_path)
        #     i_stopped = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        # else:
        #     print('No checkpoint file found!')
        #     i_stopped = 0
        i_stopped = 0

        for i in range(i_stopped, FLAGS.max_steps):
            print("Running iteration {} of TF Session".format(i))
            _, loss = sess.run([train_op, cost])
            print("The current loss is: " + str(loss))

            # Checkpoint model at each 10 iterations
            if i != 0 and i % 20 == 0 or (i+1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=i)

        encodeLayer = np.asarray(sess.run(encode))
        decodeLayer = np.asarray(sess.run(decode))

        mat2visual(encodeLayer[0, 16,:,:,:, 0], [10, 15, 19], 'encodedImage.png')
        mat2visual(decodeLayer[0, 16,:,:,:, 0], [40, 55, 60], 'decodedImage.png')





if __name__ == "__main__":
    main()
