import tensorflow as tf
import numpy as np
from utils import *
from ConvolutionalNN import *
from input_brain import *



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

    predict_op = simpleCnn.prediction(encode[0])
    return X, Y, predict_op, train_op







def main():
    X, Y, predict_op, train_op = createModel()
    print("Created the entire model! YAY!")

    # Launch the graph in a session
    with tf.Session() as sess:
    # you need to initialize all variables

        init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())

        init_op.run()
        tf.train.start_queue_runners() 

        for i in range(33):
            print("Running iteration {} of TF Session".format(i))
            sess.run(train_op)





if __name__ == "__main__":
    main()
