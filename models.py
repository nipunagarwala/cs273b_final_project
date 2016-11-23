from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import tensorflow as tf
import numpy as np
from fALFF23D import *
from ConvolutionalNN import *
from input_brain import *



def createModel():
    print("Creating the Convolutional Neural Network Object")
    simpleCnn = ConvolutionalNN()
    print("Creating the X and Y placeholder variables")
    X,Y, p_keep = simpleCnn.createVariables([None, 45, 54, 45], [None, 1])
    print("Building the CNN network")
    convProb = simpleCnn.build_simple_model(X, Y)
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
        for i in range(1000):
            print("Running the first iteration of the TF Session")
            # reducImage = sideReduction(origImage, (45,54,45), 1, (2,2,2))
            input_x, input_y = inputs(True, '/data/brain_binary_list.json', 2)
            for start, end in training_batch:
                sess.run(train_op, feed_dict={X: input_x, Y: input_y})

            test_indices = np.arange(len(teX)) # Get A Test Batch
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:test_size]




if __name__ == "__main__":
    main()
