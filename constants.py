# constants.py
# lists out hyperparameters


#############################################################################################
##																						   ##
##					                   AutoEncoder 										   ##
##																						   ##
#############################################################################################

# Auto Encoder (AE)
AE_LEARNING_RATE = 0.001
AE_HIDDEN_UNITS = [5,10,15,20]

# activation parameter for the KL Divergence cost term
AE_RHO = 0.7

# mixing term for the cost function
AE_LAMBDA = 0.6

# apply batch norm or not
AE_BATCH_NORM = True

# options for the optimizers: 'Rmsprop' 'adam' 'adagrad'
AE_OP = 'Rmsprop'

# decay rates for the optimizers
AE_BETA_1 = 0.99

# decay rate used by 'adam'
AE_BETA_2 = None

#############################################################################################
##																						   ##
##					          Convolutional AutoEncoder 								   ##
##																						   ##
#############################################################################################

# Convolutional Auto Encoder (CAE)
CAE_FILTER_SZ = [3,3]
CAE_NUM_FILTERS = [1,1]
CAE_STRIDE_SZ = [1,3]
CAE_NUM_ENCODE_LAYERS = len(CAE_FILTER_SZ)
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
MMNN_HIDDEN_UNITS = [32, 64, 32, 2]
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

CNN_NUM_FC_LAYERS = sum([1 for i in CONV_ARCH if i=='fc'])

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
CNN_LEARNING_RATE = 0.0001#0.001

# apply batch norm or not
CNN_BATCH_NORM = True

# options for the optimizers: 'Rmsprop' 'adam' 'adagrad'
CNN_OP = 'Rmsprop'

# decay rates for the optimizers
CNN_BETA_1 = 0.9

# decay rate used by 'adam'
CNN_BETA_2 = 0.2

#############################################################################################
##																						   ##
##					         Vanilla Fully Connected Neural Network						   ##
##																						   ##
#############################################################################################

# Phenotype Neural Network (NN)
# Must always end with a 1, to allow for logistic classification
NN_HIDDEN_UNITS = [32, 16, 2] # [32, 32, 32, 32, 16, 16, 16, 8, 2]

# Layer to be fed into Multi-Modal NN, 1-indexed
# Ex.
# NN_HIDDEN_UNITS = [16, 64, 64, 32, 1], NN_MMLAYER = 4
# 	-> the layer with hidden nodes 32 (the 4th layer) will be fed into MMNN
NN_MMLAYER = 2 # 4

NN_REG_CONSTANTS_WEIGHTS =[0.9]*len(NN_HIDDEN_UNITS)
NN_REG_CONSTANTS_BIAS = [0.9]*len(NN_HIDDEN_UNITS)

NN_LEARNING_RATE = 0.0001 #0.001
NN_REG_ON = False
NN_REG_OP = 'l1'

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



def create_constants_dictionary():
	const_dict = {}

	const_dict['AE_LEARNING_RATE'] = AE_LEARNING_RATE
	const_dict['AE_HIDDEN_UNITS'] = AE_HIDDEN_UNITS
	const_dict['AE_RHO'] = AE_RHO
	const_dict['AE_LAMBDA'] = AE_LAMBDA
	const_dict['AE_BATCH_NORM'] = AE_BATCH_NORM
	const_dict['AE_OP'] = AE_OP
	const_dict['AE_BETA_1'] = AE_BETA_1
	const_dict['AE_BETA_2'] = AE_BETA_2

	const_dict['CAE_FILTER_SZ'] = CAE_FILTER_SZ
	const_dict['CAE_NUM_FILTERS'] = CAE_NUM_FILTERS
	const_dict['CAE_STRIDE_SZ'] = CAE_STRIDE_SZ
	const_dict['CAE_NUM_ENCODE_LAYERS'] = CAE_NUM_ENCODE_LAYERS
	const_dict['CAE_LEARNING_RATE'] = CAE_LEARNING_RATE
	const_dict['CAE_RHO'] = CAE_RHO
	const_dict['CAE_LAMBDA'] = CAE_LAMBDA
	const_dict['CAE_BATCH_NORM'] = CAE_BATCH_NORM
	const_dict['CAE_OP'] = CAE_OP
	const_dict['CAE_BETA_1'] = CAE_BETA_1
	const_dict['CAE_BETA_2'] = CAE_BETA_2

	const_dict['MMNN_LEARNING_RATE'] = MMNN_LEARNING_RATE
	const_dict['MMNN_BATCH_NORM'] = MMNN_BATCH_NORM
	const_dict['MMNN_HIDDEN_UNITS'] = MMNN_HIDDEN_UNITS
	const_dict['MMNN_REG_CONSTANTS_WEIGHTS'] = MMNN_REG_CONSTANTS_WEIGHTS
	const_dict['MMNN_REG_CONSTANTS_BIAS'] = MMNN_REG_CONSTANTS_BIAS
	const_dict['MMNN_OP'] = MMNN_OP
	const_dict['MMNN_BETA_1'] = MMNN_BETA_1
	const_dict['MMNN_BETA_2'] = MMNN_BETA_2

	const_dict['CONV_ARCH'] = CONV_ARCH
	const_dict['CNN_NUM_LAYERS'] = CNN_NUM_LAYERS
	const_dict['CNN_NUM_FC_LAYERS'] = CNN_NUM_FC_LAYERS
	const_dict['CNN_MMLAYER'] = CNN_MMLAYER
	const_dict['CNN_REG_CONSTANTS_WEIGHTS'] = CNN_REG_CONSTANTS_WEIGHTS
	const_dict['CNN_REG_CONSTANTS_BIAS'] = CNN_REG_CONSTANTS_BIAS
	const_dict['CNN_FILTER_SZ'] = CNN_FILTER_SZ
	const_dict['CNN_NUM_FILTERS'] = CNN_NUM_FILTERS
	const_dict['CNN_STRIDE_SZ'] = CNN_STRIDE_SZ
	const_dict['CNN_POOL_SZ'] = CNN_POOL_SZ
	const_dict['CNN_POOL_STRIDE_SZ'] = CNN_POOL_STRIDE_SZ
	const_dict['CNN_REG_ON'] = CNN_REG_ON
	const_dict['CNN_REG_OP'] = CNN_REG_OP
	const_dict['CNN_RHO'] = CNN_RHO
	const_dict['CNN_LEARNING_RATE'] = CNN_LEARNING_RATE
	const_dict['CNN_BATCH_NORM'] = CNN_BATCH_NORM
	const_dict['CNN_OP'] = CNN_OP
	const_dict['CNN_BETA_1'] = CNN_BETA_1
	const_dict['CNN_BETA_2'] = CNN_BETA_2

	const_dict['NN_HIDDEN_UNITS'] = NN_HIDDEN_UNITS
	const_dict['NN_MMLAYER'] = NN_MMLAYER
	const_dict['NN_REG_CONSTANTS_WEIGHTS'] = NN_REG_CONSTANTS_WEIGHTS
	const_dict['NN_REG_CONSTANTS_BIAS'] = NN_REG_CONSTANTS_BIAS
	const_dict['NN_LEARNING_RATE'] = NN_LEARNING_RATE
	const_dict['NN_REG_ON'] = NN_REG_ON
	const_dict['NN_REG_OP'] = NN_REG_OP
	const_dict['NN_BATCH_NORM'] = NN_BATCH_NORM
	const_dict['NN_SIGMOID'] = NN_SIGMOID
	const_dict['NN_OP'] = NN_OP
	const_dict['NN_BETA_1'] = NN_BETA_1
	const_dict['NN_BETA_2'] = NN_BETA_2

	return const_dict
