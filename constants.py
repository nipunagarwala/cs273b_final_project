# constants.py
# lists out hyperparameters


#############################################################################################
##																						   ##
##					          Convolutional AutoEncoder 								   ##
##																						   ##
#############################################################################################

# Convolutional Auto Encoder (CAE)
CAE_NUM_ENCODE_LAYERS = 2
CAE_FILTER_SZ = [3,3]
CAE_NUM_FILTERS = [1,1]
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

#############################################################################################
##																						   ##
##					          Multi-Modal Neural Network								   ##
##																						   ##
#############################################################################################


# Multi-Modal NN (MMNN)
MMNN_LEARNING_RATE = 0.001

# apply batch norm or not
MMNN_BATCH_NORM = True
MMNN_HIDDEN_UNITS = [32, 64, 32, 1]
MMNN_REG_CONSTANTS_WEIGHTS = [0.6] * len(MMNN_HIDDEN_UNITS)
MMNN_REG_CONSTANTS_BIAS = [0.6] * len(MMNN_HIDDEN_UNITS)

# options for the optimizers: 'Rmsprop' 'adam' 'adagrad'
MMNN_OP = 'Rmsprop'

# decay rates for the optimizers
MMNN_BETA_1 = 0.99

# decay rate used by 'adam'
MMNN_BETA_2 = None

#############################################################################################
##																						   ##
##					          Convolutional Neural Network								   ##
##																						   ##
#############################################################################################


# Architecture of Conv Net
# Options available: 'conv' 'maxpool' 'avgpool' 'reshape' 'fc'
CONV_ARCH = ['conv', 'conv', 'maxpool', 'conv', 'conv', 'maxpool', 'conv', 'conv', 'maxpool',
		 'conv', 'conv', 'maxpool','conv', 'conv', 'maxpool','reshape', 'fc', 'fc', 'fc']

# Number of layers in the CNN, excluding the reshape layer
CNN_NUM_LAYERS = len(CONV_ARCH)

CNN_NUM_FC_LAYERS = 3

CNN_MMLAYER = CONV_ARCH.index("reshape") + 1

CNN_REG_CONSTANTS_WEIGHTS = [0.9]*CNN_NUM_LAYERS 
CNN_REG_CONSTANTS_BIAS = [0.9]*CNN_NUM_LAYERS 

CNN_FILTER_SZ = [5, 5, None, 3, 3, None, 3, 3, None, 3, 3, None, 3, 3, None, None, None, None, None]
CNN_NUM_FILTERS = [16, 32, None, 32, 32, None,32, 32, None,32, 32, None, 16, 1, None, None, None, None, None]
CNN_STRIDE_SZ = [1, 1, None, 1, 1, None, 1, 1, None,1, 1, None,1, 1, None, None, None, None, None]
CNN_POOL_SZ = [None, None, 3,None, None, 3,None, None, 3, None, None, 3, None, None, 3, None, None, None, None]
CNN_POOL_STRIDE_SZ = [None, None, 1,None, None, 1,None, None, 2, None, None, 2, None, None, 1, None, None, None, None]

# The optimizer for Regularization
# Options: 'l2' 'l1' 'kl'
# If using 'kl', kindly specify CNN_RHO too

CNN_REG_ON = True
CNN_REG_OP = 'l2'

CNN_RHO = 0.7

# Learning rate for backpropogation
CNN_LEARNING_RATE = 0.0001

# apply batch norm or not
CNN_BATCH_NORM = True

# options for the optimizers: 'Rmsprop' 'adam' 'adagrad'
CNN_OP = 'adam'

# decay rates for the optimizers
CNN_BETA_1 = 0.1

# decay rate used by 'adam'
CNN_BETA_2 = 0.01

#############################################################################################
##																						   ##
##					         Vanilla Fully Connected Neural Network						   ##
##																						   ##
#############################################################################################

# Phenotype Neural Network (NN)
# Must always end with a 1, to allow for logistic classification
NN_HIDDEN_UNITS = [32, 64, 64, 32, 1]

# Layer to be fed into Multi-Modal NN
NN_MMLAYER = 4

NN_REG_CONSTANTS_WEIGHTS = [0.6]*len(NN_HIDDEN_UNITS)
NN_REG_CONSTANTS_BIAS = [0.9]*CNN_NUM_LAYERS

NN_LEARNING_RATE = 0.01
NN_REG_ON = True
NN_REG_OP = 'l2'

# apply batch norm or not
NN_BATCH_NORM = True

# apply sigmoid or not
NN_SIGMOID = True

# options for the optimizers: 'Rmsprop' 'adam' 'adagrad'
NN_OP = 'Rmsprop'

# decay rates for the optimizers
NN_BETA_1 = 0.99

# decay rate used by 'adam'
NN_BETA_2 = None



