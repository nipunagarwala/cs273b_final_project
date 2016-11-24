import tensorflow as tf
import numpy as np
from utils import *
from ConvolutionalNN import *
from input_brain import *


batch_size = 32


def createAutoEncoderModel():
    global batch_size

    print("Creating the Convolutional AutoEncoder Object")
    simpleCnn = ConvolutionalNN()

    print("Creating the X and Y placeholder variables")
    X,Y, p_keep = simpleCnn.createVariables([batch_size, 45, 54, 45, 1], [batch_size, 45, 54, 45, 1], batch_size)

    print("Building the CNN network")
    encode,decode = simpleCnn.cnn_autoencoder(X, batch_size)

    print("Building the cost function")
    cost = simpleCnn.cost_function(decode[0], X)

    print("Building the optimization function")
    train_op = simpleCnn.minimization_function(cost, 0.001, 0.95, None, opt='Rmsprop')

    predict_op = simpleCnn.prediction(encode[0])
    return X, Y, encode[0], decode[0], cost, train_op


def createModel():
    print("Creating the Convolutional Neural Network Object")
    simpleCnn = ConvolutionalNN()

    print("Creating the X and Y placeholder variables")
    X,Y, p_keep = simpleCnn.createVariables([32, 45, 54, 45, 1], [32, 45, 54, 45, 1])
    print("Building the CNN network")
    encode,decode = simpleCnn.cnn_autoencoder(X, 32)
    # convProb, wList = simpleCnn.simple_cnn_model(X, 32)

    print("Building the cost function")
    cost = simpleCnn.cost_function(decode[0], X)

    print("Building the optimization function")
    train_op = simpleCnn.minimization_function(cost, 0.001, 0.95, None, opt='Rmsprop')

    return X, Y, encode[0], decode[0], train_op







def main():
    X, Y, encode, decode, cost, train_op = createAutoEncoderModel()
    print("Created the entire model! YAY!")

    # Launch the graph in a session
    with tf.Session() as sess:

        init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())

        init_op.run()
        tf.train.start_queue_runners() 

        for i in range(32):
            print("Running iteration {} of TF Session".format(i))
            _, loss = sess.run([train_op, cost])
            print("The current loss is: " + str(loss))

        encodeLayer = sess.run(encode)
        decodeLayer = sess.run(decode)
        print("Shape of encoded matrix: " + str(encodeLayer[16,:,:,:].shape))
        mat2visual(encodeLayer[16,:,:,:, 0], [10, 15, 19], 'encodedImage.png')
        mat2visual(decodeLayer[16,:,:,:, 0], [15, 25, 35], 'decodedImage.png')





if __name__ == "__main__":
    main()
