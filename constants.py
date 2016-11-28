# constants.py
# lists out hyperparameters

# Convolutional Auto Encoder (CAE)
CAE_FILTER_SZ = [3,3]
CAE_NUM_FILTERS = [1,1] # right now you can't change this
CAE_STRIDE_SZ = [1,3]
CAE_LEARNING_RATE = 0.001

# activation parameter for the KL Divergence cost term
CAE_RHO = 0.7
# mixing term for the cost function
CAE_LAMBDA = 0.6
# apply batch norm or not
CAE_BATCH_NORM = True

# options for the optimizers: 'Rmsprop' 'adam' 'adagrad'
CAE_OP = 'Rmsprop'
# decay rates for the optimizers
CAE_BETA_1 = 0.99
# decay rate used by 'adam'
CAE_BETA_2 = None



# Multi-Modal NN (MMNN)
MMNN_LEARNING_RATE = 0.001

# activation parameter for the KL Divergence cost term
MMNN_RHO = 0.7
# mixing term for the cost function
MMNN_LAMBDA = 0.6
# apply batch norm or not
MMNN_BATCH_NORM = True
MMNN_HIDDEN_UNITS = [32, 64, 32, 1]
MMNN_REG_CONSTANTS = [0.6] * len(MMNN_HIDDEN_UNITS)

# options for the optimizers: 'Rmsprop' 'adam' 'adagrad'
MMNN_OP = 'Rmsprop'
# decay rates for the optimizers
MMNN_BETA_1 = 0.99
# decay rate used by 'adam'
MMNN_BETA_2 = None



# Convolutional Neural Network (CNN)
CNN_NUM_LAYERS = 8
CNN_REG_CONSTANTS = [0.6]*len(NN_HIDDEN_UNITS)

CNN_LEARNING_RATE = 0.001
# apply batch norm or not
CNN_BATCH_NORM = True

# options for the optimizers: 'Rmsprop' 'adam' 'adagrad'
CNN_OP = 'Rmsprop'
# decay rates for the optimizers
CNN_BETA_1 = 0.99
# decay rate used by 'adam'
CNN_BETA_2 = None



# Phenotype Neural Network (NN)
NN_HIDDEN_UNITS = [32, 64, 64, 32, 1]
NN_REG_CONSTANTS = [0.6]*len(NN_HIDDEN_UNITS)
NN_LEARNING_RATE = 0.01
# apply batch norm or not
NN_BATCH_NORM = True

# options for the optimizers: 'Rmsprop' 'adam' 'adagrad'
NN_OP = 'Rmsprop'
# decay rates for the optimizers
NN_BETA_1 = 0.99
# decay rate used by 'adam'
NN_BETA_2 = None