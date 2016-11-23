import tensorflow as tf
import numpy as np
from fALFF23D import *
from ConvolutionalNN import *
from input_brain import *



def createModel():
    print("Creating the Convolutional Neural Network Object")
    simpleCnn = ConvolutionalNN()
    print("Creating the X and Y placeholder variables")
    X,Y, p_keep = simpleCnn.createVariables([2, 45, 54, 45, 1], [2, 45, 54, 45, 1])
    print("Building the CNN network")
    convProb = simpleCnn.cnn_autoencoder(X, Y)
    print("Building the cost function")
    cost = simpleCnn.cost_function(convProb, Y)
    print("Building the optimization function")
    train_op = simpleCnn.minimization_function(cost, 0.001, 0.9, None)
    predict_op = simpleCnn.prediction(convProb)
    return X, Y, predict_op, train_op







def main():
    X, Y, predict_op, train_op = createModel()
    print("Created the entire model! YAY!")
    # Launch the graph in a session
    with tf.Session() as sess:
    # you need to initialize all variables
        tf.initialize_all_variables().run()
        for i in range(10):
            print("Running iteration {} of TF Session".format(i))
            randX = np.random.random((2, 45, 54, 45, 1))
            yVal = np.random.random((2, 45, 54, 45, 1))
            # reducImage = sideReduction(origImage, (45,54,45), 1, (2,2,2))
            sess.run(train_op, feed_dict={X: randX, Y: yVal})




if __name__ == "__main__":
    main()
