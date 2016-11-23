from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import tensorflow as tf
import numpy as np
from fALFF import *
from ConvolutionalNN import *















def main():
    simpleCnn = ConvolutionalNN()
    X,Y, p_keep = simpleCnn.createVariables([1, 45, 54, 45], [1,1])
    convProb = simpleCnn.build_simple_model(X, Y)
    cost = simpleCnn.cost_function(convProb, Y)
    train_op = simpleCnn.minimization_function(cost, 0.001, 0.9, None)
    predict_op = simpleCnn.prediction(convProb)

    # Launch the graph in a session
    with tf.Session() as sess:
    # you need to initialize all variables
        tf.initialize_all_variables().run()
        for i in range(1000):
            origImage = loadfALFF(i)
            reducImage = sideReduction(origImage, (45,54,45), 1, (2,2,2))
            for start, end in training_batch:
                sess.run(train_op, feed_dict={X: reducImage, Y: ,
                                              p_keep_conv: 0.8, p_keep_hidden: 0.5})

            test_indices = np.arange(len(teX)) # Get A Test Batch
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:test_size]




if __name__="__main__":
    main()
