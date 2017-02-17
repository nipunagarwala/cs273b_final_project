from operator import truediv
from utils_fMRI_augment import *

import numpy as np
import csv
import itertools
import matplotlib as ml
import os
ml.use("agg")
import matplotlib.pyplot as plt
import math
import argparse
import create_brain_binaries
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from mpl_toolkits.axes_grid1 import make_axes_locatable

BRAIN_SZ = (91,109,91)
BRAIN_REGION_SZ = 116

import h5py
def write2hdf5(filename, dict2store, compression="lzf"):
    """
    Write items in a dictionary to an hdf5file

    @type   filename    :   String
    @param  filename    :   Filename of the hdf5 file to output to.
    @type   dict2store  :   Dict
    @param  dict2store  :   Dictionary of items to store. The value should be an array.

    """
    with h5py.File(filename,'w') as hf:
        for key,value in dict2store.iteritems():
            hf.create_dataset(key, data=value,compression=compression)

def hdf52dict(hdf5Filename):
    """
    Loads an HDF5 file of a game and returns a dictionary of the contents

    @type   hdf5Filename:   String
    @param  hdf5Filename:   Filename of the hdf5 file.
    """
    retDict = {}
    with h5py.File(hdf5Filename,'r') as hf:
        for key in hf.keys():
            retDict[key] = np.array(hf.get(key))

    return retDict

def print_hdf5Files(h5filename):
    with h5py.File(h5filename,'r') as hf:
        for key in hf.keys():
            print '-'*20
            print key
            chil = hf.get(key)
            print chil
            for key2 in chil:
                print key2
                chil2 = chil.get(key2)
                print chil2
                print chil2.value

#print_hdf5Files('example2_weights.hdf5')

def saveWeights(metafile, chkptfile, h5filename, weight_names_tf):
    # load the checkpointed session
    sess = tf.Session()
    new_saver = tf.train.import_meta_graph(metafile)
    new_saver.restore(sess, chkptfile)

    tf_weight_vals = {}
    all_vars = tf.trainable_variables()
    for v in all_vars:
        tf_weight_vals[v.name] = sess.run(v)

    print tf_weight_vals.keys()

    f = h5py.File(h5filename, 'w')
    for layerIndx in range(len(weight_names_tf)):
        g = f.create_group('layer'+str(layerIndx))

        weight_names = []
        weight_values = []
        numVars = len(weight_names_tf[layerIndx])
        for i in range(numVars):
            name = 'param_' + str(i)
            weight_names.append(name.encode('utf8'))
            weight_values.append(tf_weight_vals[weight_names_tf[layerIndx][i]])
        g.attrs['weight_names'] = weight_names

        for name, val in zip(weight_names, weight_values):
            param_dset = g.create_dataset(name, val.shape,
                                          dtype=val.dtype)
            if not val.shape:
                # scalar
                param_dset[()] = val
            else:
                param_dset[:] = val

    f.flush()
    f.close()

# metafile = './chkpt/model.ckpt-99.meta'
# chkptfile = './chkpt/model.ckpt-99'
# h5filename = 'a.hdf5'
# weight_names_tf = [('layer1_filters:0','Variable:0'),('layer2_filters:0','Variable_1:0'),()]
# saveWeights(metafile,chkptfile,h5filename,weight_names_tf)
# print_hdf5Files(h5filename)

def brainRegion2brainID(brainRegions):
    """
    Converts brainRegions between 1 and 116 to brainID specified in /data/region_name.csv
    """
    if type(brainRegions)==int:
        brainRegions = [brainRegions]

    assert not 0 in brainRegions, "Brain Region is 1 indexed!"

    brainRegions.append(0)
    brainRegions = sorted(brainRegions)

    brainIDs = []
    with open('/data/region_name.csv') as csvfile:
        csvR = csv.reader(csvfile)
        next(csvR)
        for i in range(len(brainRegions)-1):
            for j in range(brainRegions[i+1]-brainRegions[i]-1):
                next(csvR)
            brainIDs.append(int(next(csvR)[2]))

    brainIDs = sorted(brainIDs)

    return brainIDs

def saveBrainRegion2npy():
    import pickle
    brainMat = np.zeros(BRAIN_SZ, dtype=int)-1

    id2regionDict = {}
    with open('/data/region_name.csv') as csvfile:
        csvR = csv.reader(csvfile)
        next(csvR)
        for row in csvR:
            id2regionDict[int(row[2])] = int(row[1])-1

    with open('/data/coord.csv') as csvfile:
        csvR = csv.reader(csvfile)
        coordsList = list(csvR)
        coords = np.array(coordsList)
        coords = coords[1:,1:]
        coords = coords.astype('int')-1

    brainRegionDict = {}
    convertedBrainRegionDict = {}
    with open('/data/region_code.csv') as csvfile:
        csvR = csv.reader(csvfile)
        next(csvR)

        count = 0
        for row in csvR:
            coord = coords[count,:]
            rawID = int(row[1])
            brainMat[coord[0],coord[1],coord[2]] = id2regionDict[rawID]
            if rawID not in brainRegionDict:
                brainRegionDict[rawID] = []
                convertedBrainRegionDict[id2regionDict[rawID]+1] = []
            brainRegionDict[rawID].append((coord[0],coord[1],coord[2]))
            convertedBrainRegionDict[id2regionDict[rawID]+1].append((coord[0],coord[1],coord[2]))
            count += 1

    np.save('/data/useful_npy/index2BrainRegion',brainMat)
    pickle.dump(brainRegionDict, open('/data/useful_npy/rawBrainRegionID2Coords.p','wb'))
    pickle.dump(convertedBrainRegionDict, open('/data/useful_npy/brainRegionID2Coords.p','wb'))

def weights2Brain(weights):
    """
    Constructs a 3D brain image from 'weights', which represents the value assigned for
    each region of a brain.

    @type   weights     :   3D numpy matrix
    @param  weights     :   Values to assign for each brain region

    @rtype  brainMat    :   3D numpy matrix
    @return brainMat    :   A brain with each brain regions with the values
                            specified in 'weights'
    """
    brainMat = np.zeros(BRAIN_SZ)
    index2region = np.load('/data/useful_npy/index2BrainRegion.npy')

    for x in range(BRAIN_SZ[0]):
        for y in range(BRAIN_SZ[1]):
            for z in range(BRAIN_SZ[2]):
                if index2region[x,y,z]!=-1:
                    brainMat[x,y,z] = weights[index2region[x,y,z]]

    return brainMat

def mat2visual(mat, zLocs, filename, valRange='auto'):
    """
    Visualizes the input numpy matrix and saves it into a file

    Example Usage:
        data = loadfALFF(4)
        mat2visual(data, [40,45,50], 'example.png')

    @type   mat         :   3D numpy matrix
    @param  mat         :   3D data matrix to visualize
    @type   zLocs       :   int array
    @param  zLocs       :   Specifies the z positions to slice the mat matrix at
    @type   filename    :   String
    @param  filename    :   Name of the file to save the sliced brains to.
    @type   valRange    :   int tuple or 'auto'
    @param  valRange    :   Specifies the maximum and minimum values of the colorbar used in imshow.
                            'auto' for auto-scaling of the input
    """
    _,_,c = mat.shape
    plt.close("all")
    plt.figure()
    for i in range(len(zLocs)):
        if zLocs[i]>=c:
            print("An element %d in zLocs is larger than %d" %(zLocs[i],c))
            return
        plt.subplot(1,len(zLocs),i+1)
        plt.title('z='+str(zLocs[i]))
        if type(valRange) is str and valRange=='auto':
            # plt.imshow(mat[:,:,zLocs[i]], cmap = "gray", interpolation='none')
            plt.imshow(mat[:,:,zLocs[i]], interpolation='none')
        else:
            # plt.imshow(mat[:,:,zLocs[i]], vmin=min(valRange), vmax=max(valRange), cmap = "gray", interpolation='none')
            plt.imshow(mat[:,:,zLocs[i]], vmin=min(valRange), vmax=max(valRange), interpolation='none')

    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cax=cax)

    plt.savefig(filename)

# this doesn't work qq
def makeAnimatedGif(path, filename):
    from images2gif import writeGif
    from PIL import Image
    # Recursively list image files and store them in a variable
    imgs = sorted(os.listdir(path))
    images = [Image.open(os.path.join(path,i)) for i in imgs]

    writeGif(filename, images, duration=0.1)
    print os.path.realpath(filename)
    print "%s has been created" % filename

def coolPics():
    # # blackout brains
    # mat = loadfALFF(3)
    # mat2 = blackOutBrain(mat,[1,2])
    # mat2visual(mat, [20,40,60], 'original.png')
    # mat2visual(mat2, [20,40,60], 'blackedOut.png')

    # # brain_regions.png
    # import random
    # weights = [random.randint(0,500)+300 for i in range(116)]
    # mat = weights2Brain(weights)
    # mat2visual(mat, [20,40,60], 'brain_regions.png', [0,1000])

    pass

import copy
import itertools
import multiprocessing as mp

CLASS_NUM = 2

# adapted from https://github.com/openai/cleverhans/tree/5c6ece85ffe82441a5512b4ff4120fd904aedab4

def jacobian(sess, x, grads, label, X, phase_train):
    """
    TensorFlow implementation of the foward derivative / Jacobian
    :param x: the input placeholder
    :param grads: the list of TF gradients returned by jacobian_graph()
    :param label: the label
    :param X: numpy array with sample input
    :return: matrix of forward derivatives flattened into vectors
    """
    # Prepare feeding dictionary for all gradient computations
    feed_dict = {x: X, phase_train: False}

    # Initialize a numpy array to hold the Jacobian component values
    jacobian_val = np.zeros((CLASS_NUM, X.shape[0], X.shape[1], X.shape[2], X.shape[3], X.shape[4]), dtype=np.float32)

    print(X.shape)
    # Compute the gradients for all classes
    for class_ind, grad in enumerate(grads):
        jacobian_val[class_ind, :, :, :, :] = sess.run(grad, feed_dict)

    return jacobian_val


def jacobian_graph(predictions, x):
    """
    Create the Jacobian graph to be ran later in a TF session
    :param predictions: the model's symbolic output (linear output, pre-softmax)
    :param x: the input placeholder
    :return:
    """
    # This function will return a list of TF gradients
    list_derivatives = []

    # Define the TF graph elements to compute our derivatives for each class
    for class_ind in xrange(CLASS_NUM):
        derivatives, = tf.gradients(predictions[:, class_ind], x)
        list_derivatives.append(derivatives)

    return list_derivatives

def saliency_tf(metafile, chkptfile, sample, label):
    """
    TensorFlow implementation of the JSMA (see https://arxiv.org/abs/1511.07528
    for details about the algorithm design choices).
    :param metafile     : a metafile for the TF session
    :param chkptfile    : a checkpoint file for the TF session
    :param sample       : numpy array with sample input
    :param label        : label for sample input
    :return             : output saliency map
    """
    # restore a session from the provided files
    sess = tf.Session()
    new_saver = tf.train.import_meta_graph(metafile)
    new_saver.restore(sess, chkptfile)

    # x: the input placeholder
    x = tf.get_variable('input_placeholder')

    # predictions: the model's symbolic output (linear output, pre-softmax)
    predictions = tf.get_variable('pre_softmax')

    # connect the gradients from the input to the output
    grads = jacobian_graph(predictions, x)

    adv_x = copy.copy(sample)
    # Compute the Jacobian components
    grad_vals = jacobian(sess, x, grads, label, adv_x)

    # visualize the result
    mat2visual(grad_vals[0], [20,40,60], 'control.png')
    mat2visual(grad_vals[1], [20,40,60], 'autistic.png')

def blackOutVisualization(probs, filenames):
    import re
    patientDic = {}

    for fname,prob in zip(filenames,probs):
        patientID = int(re.search('[0-9]+',re.search('compressed_[0-9]+',fname).group(0)).group(0))
        if patientID not in patientDic:
            patientDic[patientID] = [0]*117

        if 'original' in fname:
            regionID = 116
        else:
            regionID = int(re.search('[0-9]+',re.search('blackout_[0-9]+',fname).group(0)).group(0))
        patientDic[patientID][regionID] = prob

    autID,ctlID = getGroupLabels()
    label = ''
    for p in patientDic:
        patientProbs = patientDic[p]
        if p in autID:
            weights = [patientProbs[116]-pr for pr in patientProbs]
            label = 'autistic'
        else:
            weights = [pr-patientProbs[116] for pr in patientProbs]
            label = 'control'
        np.save('blackout_%d_%s'%(p,label), weights2Brain(weights))
        mat2visual(np.load('blackout_%d_%s.npy'%(p,label)),[20,40,60],'%d.png'%p)

if __name__ == '__main__':
    # probs = [0.7016589, 0.53661954, 0.038011093, 0.095822118, 0.36838549, 0.54734296, 0.82986736, 0.067482933, 0.43189046, 0.29399946, 0.97235143, 0.9984926, 0.15775973, 0.87002844, 0.34289947, 0.063488521, 0.4185423, 0.6518721, 0.99747366, 0.058898017, 0.99871695, 0.93766397, 0.056865543, 0.49278185, 0.29352495, 0.36201373, 0.10308384, 0.59026277, 0.15358131, 0.1050192, 0.17410569, 0.96552068, 0.13877985, 0.44763544, 0.15078872, 0.57347137, 0.99842632, 0.86192364, 0.024976781, 0.9967339, 0.37870184, 0.19819644, 0.94992894, 0.87529212, 0.80342573, 0.6170997, 0.90802497, 0.020024313, 0.99612832, 0.017418377, 0.47476119, 0.93847084, 0.014785589, 0.85475326, 0.12937762, 0.27303058, 0.65717304, 0.8773132, 0.012717852, 0.61772919, 0.60490435, 0.74976897, 0.10774954, 0.74683213, 0.95324808, 0.2183966, 0.88531643, 0.20251949, 0.13436472, 0.014455118, 0.98326981, 0.63705248, 0.98977637, 0.90120858, 0.3041876, 0.066700965, 0.97961754, 0.11409438, 0.98929751, 0.1681513, 0.26058024, 0.98154217, 0.053559862, 0.016572911, 0.029982191, 0.086816601, 0.96883863, 0.031379167, 0.23783676, 0.41104996, 0.2604031, 0.17460622, 0.52030987, 0.60456687, 0.019428149, 0.61032277, 0.46552974, 0.96263915, 0.32996598, 0.1567201, 0.87029815, 0.4064301, 0.99441916, 0.38567215, 0.97992939, 0.010425194, 0.050834868, 0.003963667, 0.96696633, 0.40417117, 0.77552205, 0.51984763, 0.80552757, 0.043172807, 0.97509909, 0.52099466, 0.71058291, 0.2037373, 0.69937944, 0.54766983, 0.0058306083, 0.81456715, 0.87895459, 0.0084501076, 0.66412187, 0.0020940867, 0.14599629, 0.059255451]
    # filenames = ['/data/blackoutVisual_reduced/compressed_157_blackout_90_autistic_visual_64.bin:0', '/data/blackoutVisual_reduced/compressed_367_blackout_52_control_visual_65.bin:0', '/data/blackoutVisual_reduced/compressed_780_blackout_9_control_visual_66.bin:0', '/data/blackoutVisual_reduced/compressed_27_blackout_109_control_visual_67.bin:0', '/data/blackoutVisual_reduced/compressed_367_blackout_49_control_visual_68.bin:0', '/data/blackoutVisual_reduced/compressed_780_blackout_15_control_visual_69.bin:0', '/data/blackoutVisual_reduced/compressed_780_blackout_100_control_visual_70.bin:0', '/data/blackoutVisual_reduced/compressed_27_blackout_21_control_visual_71.bin:0', '/data/blackoutVisual_reduced/compressed_478_blackout_93_autistic_visual_72.bin:0', '/data/blackoutVisual_reduced/compressed_780_blackout_52_control_visual_73.bin:0', '/data/blackoutVisual_reduced/compressed_880_blackout_104_autistic_visual_74.bin:0', '/data/blackoutVisual_reduced/compressed_880_blackout_102_autistic_visual_75.bin:0', '/data/blackoutVisual_reduced/compressed_478_blackout_59_autistic_visual_76.bin:0', '/data/blackoutVisual_reduced/compressed_880_blackout_28_autistic_visual_77.bin:0', '/data/blackoutVisual_reduced/compressed_880_blackout_5_autistic_visual_78.bin:0', '/data/blackoutVisual_reduced/compressed_3_blackout_45_autistic_visual_79.bin:0', '/data/blackoutVisual_reduced/compressed_157_blackout_74_autistic_visual_80.bin:0', '/data/blackoutVisual_reduced/compressed_908_blackout_67_control_visual_81.bin:0', '/data/blackoutVisual_reduced/compressed_780_blackout_27_control_visual_82.bin:0', '/data/blackoutVisual_reduced/compressed_475_blackout_17_autistic_visual_83.bin:0', '/data/blackoutVisual_reduced/compressed_27_blackout_79_control_visual_84.bin:0', '/data/blackoutVisual_reduced/compressed_367_blackout_45_control_visual_85.bin:0', '/data/blackoutVisual_reduced/compressed_3_blackout_49_autistic_visual_86.bin:0', '/data/blackoutVisual_reduced/compressed_908_blackout_59_control_visual_87.bin:0', '/data/blackoutVisual_reduced/compressed_780_blackout_108_control_visual_88.bin:0', '/data/blackoutVisual_reduced/compressed_367_blackout_78_control_visual_89.bin:0', '/data/blackoutVisual_reduced/compressed_478_blackout_98_autistic_visual_90.bin:0', '/data/blackoutVisual_reduced/compressed_27_blackout_31_control_visual_91.bin:0', '/data/blackoutVisual_reduced/compressed_27_blackout_94_control_visual_92.bin:0', '/data/blackoutVisual_reduced/compressed_478_blackout_13_autistic_visual_93.bin:0', '/data/blackoutVisual_reduced/compressed_367_blackout_73_control_visual_94.bin:0', '/data/blackoutVisual_reduced/compressed_908_blackout_39_control_visual_95.bin:0', '/data/blackoutVisual_reduced/compressed_475_blackout_original_autistic_visual_128.bin:0', '/data/blackoutVisual_reduced/compressed_780_blackout_89_control_visual_129.bin:0', '/data/blackoutVisual_reduced/compressed_780_blackout_53_control_visual_130.bin:0', '/data/blackoutVisual_reduced/compressed_478_blackout_16_autistic_visual_131.bin:0', '/data/blackoutVisual_reduced/compressed_367_blackout_42_control_visual_132.bin:0', '/data/blackoutVisual_reduced/compressed_367_blackout_71_control_visual_133.bin:0', '/data/blackoutVisual_reduced/compressed_880_blackout_74_autistic_visual_134.bin:0', '/data/blackoutVisual_reduced/compressed_3_blackout_9_autistic_visual_135.bin:0', '/data/blackoutVisual_reduced/compressed_367_blackout_67_control_visual_136.bin:0', '/data/blackoutVisual_reduced/compressed_908_blackout_9_control_visual_137.bin:0', '/data/blackoutVisual_reduced/compressed_157_blackout_31_autistic_visual_138.bin:0', '/data/blackoutVisual_reduced/compressed_27_blackout_26_control_visual_139.bin:0', '/data/blackoutVisual_reduced/compressed_157_blackout_58_autistic_visual_140.bin:0', '/data/blackoutVisual_reduced/compressed_27_blackout_115_control_visual_141.bin:0', '/data/blackoutVisual_reduced/compressed_478_blackout_49_autistic_visual_142.bin:0', '/data/blackoutVisual_reduced/compressed_478_blackout_109_autistic_visual_143.bin:0', '/data/blackoutVisual_reduced/compressed_367_blackout_113_control_visual_144.bin:0', '/data/blackoutVisual_reduced/compressed_27_blackout_59_control_visual_145.bin:0', '/data/blackoutVisual_reduced/compressed_3_blackout_116_autistic_visual_146.bin:0', '/data/blackoutVisual_reduced/compressed_908_blackout_108_control_visual_147.bin:0', '/data/blackoutVisual_reduced/compressed_780_blackout_94_control_visual_148.bin:0', '/data/blackoutVisual_reduced/compressed_367_blackout_82_control_visual_149.bin:0', '/data/blackoutVisual_reduced/compressed_908_blackout_62_control_visual_150.bin:0', '/data/blackoutVisual_reduced/compressed_880_blackout_67_autistic_visual_151.bin:0', '/data/blackoutVisual_reduced/compressed_478_blackout_46_autistic_visual_152.bin:0', '/data/blackoutVisual_reduced/compressed_780_blackout_111_control_visual_153.bin:0', '/data/blackoutVisual_reduced/compressed_475_blackout_70_autistic_visual_154.bin:0', '/data/blackoutVisual_reduced/compressed_880_blackout_47_autistic_visual_155.bin:0', '/data/blackoutVisual_reduced/compressed_157_blackout_107_autistic_visual_156.bin:0', '/data/blackoutVisual_reduced/compressed_367_blackout_76_control_visual_157.bin:0', '/data/blackoutVisual_reduced/compressed_3_blackout_77_autistic_visual_158.bin:0', '/data/blackoutVisual_reduced/compressed_908_blackout_88_control_visual_159.bin:0', '/data/blackoutVisual_reduced/compressed_478_blackout_30_autistic_visual_192.bin:0', '/data/blackoutVisual_reduced/compressed_908_blackout_103_control_visual_193.bin:0', '/data/blackoutVisual_reduced/compressed_27_blackout_95_control_visual_194.bin:0', '/data/blackoutVisual_reduced/compressed_880_blackout_56_autistic_visual_195.bin:0', '/data/blackoutVisual_reduced/compressed_908_blackout_85_control_visual_196.bin:0', '/data/blackoutVisual_reduced/compressed_780_blackout_13_control_visual_197.bin:0', '/data/blackoutVisual_reduced/compressed_27_blackout_89_control_visual_198.bin:0', '/data/blackoutVisual_reduced/compressed_157_blackout_63_autistic_visual_199.bin:0', '/data/blackoutVisual_reduced/compressed_27_blackout_56_control_visual_200.bin:0', '/data/blackoutVisual_reduced/compressed_478_blackout_9_autistic_visual_201.bin:0', '/data/blackoutVisual_reduced/compressed_880_blackout_90_autistic_visual_202.bin:0', '/data/blackoutVisual_reduced/compressed_475_blackout_94_autistic_visual_203.bin:0', '/data/blackoutVisual_reduced/compressed_157_blackout_88_autistic_visual_204.bin:0', '/data/blackoutVisual_reduced/compressed_3_blackout_17_autistic_visual_205.bin:0', '/data/blackoutVisual_reduced/compressed_3_blackout_94_autistic_visual_206.bin:0', '/data/blackoutVisual_reduced/compressed_157_blackout_103_autistic_visual_207.bin:0', '/data/blackoutVisual_reduced/compressed_27_blackout_43_control_visual_208.bin:0', '/data/blackoutVisual_reduced/compressed_475_blackout_60_autistic_visual_209.bin:0', '/data/blackoutVisual_reduced/compressed_780_blackout_92_control_visual_210.bin:0', '/data/blackoutVisual_reduced/compressed_908_blackout_15_control_visual_211.bin:0', '/data/blackoutVisual_reduced/compressed_27_blackout_113_control_visual_212.bin:0', '/data/blackoutVisual_reduced/compressed_478_blackout_2_autistic_visual_213.bin:0', '/data/blackoutVisual_reduced/compressed_880_blackout_38_autistic_visual_214.bin:0', '/data/blackoutVisual_reduced/compressed_3_blackout_5_autistic_visual_215.bin:0', '/data/blackoutVisual_reduced/compressed_880_blackout_82_autistic_visual_216.bin:0', '/data/blackoutVisual_reduced/compressed_3_blackout_8_autistic_visual_217.bin:0', '/data/blackoutVisual_reduced/compressed_880_blackout_116_autistic_visual_218.bin:0', '/data/blackoutVisual_reduced/compressed_908_blackout_76_control_visual_219.bin:0', '/data/blackoutVisual_reduced/compressed_3_blackout_103_autistic_visual_220.bin:0', '/data/blackoutVisual_reduced/compressed_475_blackout_80_autistic_visual_221.bin:0', '/data/blackoutVisual_reduced/compressed_157_blackout_53_autistic_visual_222.bin:0', '/data/blackoutVisual_reduced/compressed_780_blackout_7_control_visual_223.bin:0', '/data/blackoutVisual_reduced/compressed_780_blackout_73_control_visual_256.bin:0', '/data/blackoutVisual_reduced/compressed_780_blackout_110_control_visual_257.bin:0', '/data/blackoutVisual_reduced/compressed_478_blackout_11_autistic_visual_258.bin:0', '/data/blackoutVisual_reduced/compressed_880_blackout_33_autistic_visual_259.bin:0', '/data/blackoutVisual_reduced/compressed_157_blackout_77_autistic_visual_260.bin:0', '/data/blackoutVisual_reduced/compressed_478_blackout_88_autistic_visual_261.bin:0', '/data/blackoutVisual_reduced/compressed_880_blackout_10_autistic_visual_262.bin:0', '/data/blackoutVisual_reduced/compressed_367_blackout_90_control_visual_263.bin:0', '/data/blackoutVisual_reduced/compressed_3_blackout_38_autistic_visual_264.bin:0', '/data/blackoutVisual_reduced/compressed_880_blackout_39_autistic_visual_265.bin:0', '/data/blackoutVisual_reduced/compressed_880_blackout_17_autistic_visual_266.bin:0', '/data/blackoutVisual_reduced/compressed_880_blackout_96_autistic_visual_267.bin:0', '/data/blackoutVisual_reduced/compressed_367_blackout_12_control_visual_268.bin:0', '/data/blackoutVisual_reduced/compressed_478_blackout_69_autistic_visual_269.bin:0', '/data/blackoutVisual_reduced/compressed_3_blackout_21_autistic_visual_270.bin:0', '/data/blackoutVisual_reduced/compressed_367_blackout_1_control_visual_271.bin:0', '/data/blackoutVisual_reduced/compressed_367_blackout_89_control_visual_272.bin:0', '/data/blackoutVisual_reduced/compressed_3_blackout_55_autistic_visual_273.bin:0', '/data/blackoutVisual_reduced/compressed_880_blackout_108_autistic_visual_274.bin:0', '/data/blackoutVisual_reduced/compressed_475_blackout_6_autistic_visual_275.bin:0', '/data/blackoutVisual_reduced/compressed_157_blackout_10_autistic_visual_276.bin:0', '/data/blackoutVisual_reduced/compressed_27_blackout_44_control_visual_277.bin:0', '/data/blackoutVisual_reduced/compressed_157_blackout_4_autistic_visual_278.bin:0', '/data/blackoutVisual_reduced/compressed_908_blackout_32_control_visual_279.bin:0', '/data/blackoutVisual_reduced/compressed_157_blackout_61_autistic_visual_280.bin:0', '/data/blackoutVisual_reduced/compressed_157_blackout_41_autistic_visual_281.bin:0', '/data/blackoutVisual_reduced/compressed_367_blackout_92_control_visual_282.bin:0', '/data/blackoutVisual_reduced/compressed_880_blackout_66_autistic_visual_283.bin:0', '/data/blackoutVisual_reduced/compressed_367_blackout_109_control_visual_284.bin:0', '/data/blackoutVisual_reduced/compressed_908_blackout_31_control_visual_285.bin:0', '/data/blackoutVisual_reduced/compressed_3_blackout_30_autistic_visual_286.bin:0', '/data/blackoutVisual_reduced/compressed_157_blackout_23_autistic_visual_287.bin:0']
    # blackOutVisualization(probs,filenames)
    pass