import tensorflow as tf
import numpy as np
from utils import *
from ConvolutionalNN import *
from models import *
from input_brain import *


batch_size = 32


def createAutoEncoderModel():
    global batch_size

    print("Creating the Convolutional AutoEncoder Object")
    cae = ConvAutoEncoder([batch_size, 45, 54, 45, 1], [batch_size, 45, 54, 45, 1], batch_size, 0.001, 0.99, None, op='Rmsprop')

    allFilters = [[3, 3, 3, 1, 1],[3, 3, 3, 1, 1]]
    allStrides = [[1, 1, 1, 1, 1],[1, 2, 2, 2, 1]]
    allNames = ["layer1_filters","layer2_filters","layer3_filters","layer4_filters","layer5_filters","layer6_filters",
                "layer7_filters","layer8_filters","layer9_filters","layer10_filters"]
    allRelu = [True]*4
    allBatch = [False]*4

    # We do not need ReLUs in the encoder layer and the decode layer
    allRelu[1] = False
    allRelu[3] = False
    allBatch[1] = False
    allBatch[3] = False
    layer_outputs, weights, weight_shapes, encode, decode = cae.build_model(2, allFilters, allStrides, allNames, allRelu, allBatch)

    print("Setting up the Training model of the Autoencoder")
    cost, train_op = cae.train()
    return layer_outputs, weights, weight_shapes, encode, decode, cost, train_op


    


def createModel():
    print("Creating the Convolutional Neural Network Object")
    simpleCnn = CNNLayers()

    print("Creating the X and Y placeholder variables")
    X,Y, p_keep = simpleCnn.createVariables([32, 45, 54, 45, 1], [32, 45, 54, 45, 1], 32)
    print("Building the CNN network")
    encode,decode, weightList = simpleCnn.cnn_autoencoder(X, 32)
    # convProb, wList = simpleCnn.simple_cnn_model(X, 32)

    print("Building the cost function")
    cost = simpleCnn.cost_function(decode[0], X)

    print("Building the optimization function")
    train_op = simpleCnn.minimization_function(cost, 0.001, 0.95, None, opt='Rmsprop')

    return X, Y, encode[0], decode[0], cost, train_op







def main():
    layer_outputs, weights, weight_shapes, encode, decode, cost, train_op = createAutoEncoderModel()
    # X, Y, encode, decode, cost, train_op = createModel()
    print("Created the entire model! YAY!")

    # Launch the graph in a session
    with tf.Session() as sess:

        init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())

        init_op.run()
        tf.train.start_queue_runners() 

        for i in range(100):
            print("Running iteration {} of TF Session".format(i))
            _, loss = sess.run([train_op, cost])
            print("The current loss is: " + str(loss))

        encodeLayer = np.asarray(sess.run(encode))
        decodeLayer = np.asarray(sess.run(decode))
        print("Shape of encodeLayer: "+ str(encodeLayer))
        print(type(encodeLayer))
        print(encodeLayer.shape)
        mat2visual(encodeLayer[0, 16,:,:,:, 0], [10, 15, 19], 'encodedImage.png')
        mat2visual(decodeLayer[0, 16,:,:,:, 0], [40, 55, 60], 'decodedImage.png')





if __name__ == "__main__":
    main()
