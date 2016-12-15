import tensorflow as tf
from graph_saver_api import *
import numpy as np
from utils_fMRI_augment import *



def saliency(image, scores):
	LABEL_SZ = 1
	PHENO_SZ = 29
	X_SZ = 31
	Y_SZ = 37
	Z_SZ = 31
	batchSz = 32
	autismLabels, nonAutLabels = getGroupLabels()
	brainPartialPath = '/data/binaries_reduced2/compressed_'

	incDimBrain = np.zeros((batchSz, X_SZ, Y_SZ, Z_SZ, 1))
	incLabels = np.zeros((batchSz, 2))
	for i in range(32):
		brainPath = brainPartialPath + str(autismLabels[i]) + '.bin'
		incLabels[i,1] = 1
		inputBrain = np.memmap(filename=brainPath, dtype='float32',
	                          mode='r', offset=(LABEL_SZ+PHENO_SZ)*4, shape=(X_SZ,Y_SZ,Z_SZ))
		incDimBrain[i,:,:,:,0] = inputBrain


	# batch_size = 16

	# # get the session ready

	# # We can verify that we can access to the list of operations in the graph
	# for op in graph.get_operations():
	# 	# if 'Softmax' in op.name:
	# 	print(op.name)
	# 	print(op.values())

	# # We access the input and output nodes 
	# x = graph.get_tensor_by_name('prefix/shuffle_batch:0')
	# y = graph.get_tensor_by_name('prefix/Add_12:0')

	x = image
	y = scores

	grads = jacobian_graph(y, x)

	adv_x = copy.copy(x)
    # Compute the Jacobian components
	grad_vals = jacobian(sess, x, grads, incLabels, incDimBrain)

	# y_out = sess.run(y, feed_dict={
	# 	x: curBatch # < 45
	# })

    # visualize the result
    # mat2visual(grad_vals[0], [20,40,60], 'control.png')
	print(np.max(grad_vals[1,0,:,:,:,0]))
	mat2visual(grad_vals[1,0,:,:,:,0], [10,15,25], 'autistic1.png')
	mat2visual(grad_vals[0,0,:,:,:,0], [10,15,25], 'autistic2.png')
	mat2visual(grad_vals[1,1,:,:,:,0], [10,15,25], 'autistic3.png')
	mat2visual(grad_vals[1,2,:,:,:,0], [10,15,25], 'autistic4.png')
	mat2visual(grad_vals[1,3,:,:,:,0], [10,15,25], 'autistic5.png')