import numpy as np
# from utils_fMRI_augment import *
from utils_visual import *



def saliency(image, scores, sess, phase_train):
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
        brainPath = brainPartialPath + str(i+1) + '.bin'
        if i in autismLabels:
            incLabels[i,1] = 1
        else:
            incLabels[i,0] = 1
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
    grad_vals = jacobian(sess, x, grads, incLabels, incDimBrain, phase_train)

    # y_out = sess.run(y, feed_dict={
    # 	x: curBatch # < 45
    # })

    # visualize the result
    # mat2visual(grad_vals[0], [20,40,60], 'control.png')
    # print(np.max(grad_vals[1,0,:,:,:,0]))
    mat2visual(grad_vals[1,0,:,:,:,0], [10,15,25], 'autistic1.png', [-0.35, 0.35])
    mat2visual(grad_vals[1,1,:,:,:,0], [10,15,25], 'autistic3.png', [-0.35, 0.35])
    mat2visual(grad_vals[0,26,:,:,:,0], [10,15,25], 'autistic2.png', [-0.35, 0.35])
    mat2visual(grad_vals[0,2,:,:,:,0], [10,15,25], 'autistic4.png', [-0.35, 0.35])
    mat2visual(grad_vals[1,3,:,:,:,0], [10,15,25], 'autistic5.png', [-0.35, 0.35])

    for j in range(32):
        for k in range(2):
            name = "_autism_" if incLabels[j,1] == 1 else "_control_"
            fullfile = str(j+1) + name + str(k)
            np.save("/data/saliency/" + fullfile, grad_vals[k,j,:,:,:,0])
>>>>>>> visuals
