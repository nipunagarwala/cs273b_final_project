from operator import truediv

import numpy as np
import csv
import itertools
import matplotlib as ml
import os
ml.use("agg")
import matplotlib.pyplot as plt
import math

from mpl_toolkits.axes_grid1 import make_axes_locatable

BRAIN_SZ = (91,109,91)
BRAIN_REGION_SZ = 116


def loadfALFF(patientID):
    """
    Converts the fALFF brain readings to a 3D representation in numpy.
    The coordinates should be specified in the coord.csv.

    @type   patientID   :   int
    @param  patientID   :   id of the patient to load the brain. 1 indexed!

    @rtype  threeDMat   :   3D numpy matrix
    @return threeDMat   :   Stores information about the fALFF brain readings
                            in 3D representation, with empty portions 0-padded
    """
    assert not patientID==0, "Patient ID is 1 indexed!"

    threeDMat = np.zeros(BRAIN_SZ)

    with open('/data/coord.csv') as csvfile:
        csvR = csv.reader(csvfile)
        coordsList = list(csvR)
        coords = np.array(coordsList)
        coords = coords[1:,1:]
        coords = coords.astype('int')-1


    with open('/data/processed_phenotype_data.csv') as csvfile:
        id = next(itertools.islice(csv.reader(csvfile), patientID, None))[2]

    count = 0
    fileName = '/data/fALFF_Data/'+str(id)+'.csv'
    with open(fileName) as csvfile:
        csvR = csv.reader(csvfile)
        first = True
        for row in csvR:
            if first:
                first = False
                continue

            coord = coords[count,:]
            threeDMat[coord[0],coord[1],coord[2]] = float(row[1])
            count += 1

    return threeDMat

def sizeReduction(data, targetShape, opt, poolBox=(2,2,2), filename=None):
    """
    Reduces the dimensionality of the 3D data stored in data to targetShape

    Example Usage:
        data = loadfALFF(4)
        avgPoolData = sizeReduction(data, (45, 54, 45), opt=1, poolBox=(2,2,2))
        randomPollData1 = sizeReduction(data, (45, 54, 45), opt=2, poolBox=(2,2,2))

    @type   data        :   3D numpy matrix
    @param  data        :   3D data matrix to dimension reduce from
    @type   targetShape :   3 element int list
    @param  targetShape :   Specifies the size of the output matrix
    @type   opt         :   int
    @param  opt         :   Dimension reduction methodology.
                            opt=0: take the value in the middle
                            opt=1: average pooling
                            opt=2: random pooling
                            opt=3: max pooling
    @type   poolBox     :   3 element int list
    @param  poolBox     :   Size of the box to "pool" from
    @type   filename    :   String
    @param  filename    :   Name of the file to save the reduced matrix to.
                            The saved file can be loaded with np.load()

    @rtype  threeDMatRedux:   3D numpy matrix
    @return threeDMatRedux:   Reduced version of the input data
    """
    ratio = map(truediv,BRAIN_SZ,targetShape)
    threeDMatRedux = np.zeros(targetShape)

    lowBoxExtension = [int(math.floor(i/2.0)) for i in poolBox]
    highBoxExtension = [int(math.ceil(i/2.0)) for i in poolBox]

    for x in range(targetShape[0]):
        xDisc = int(math.floor(x*ratio[0]))
        for y in range(targetShape[1]):
            yDisc = int(math.floor(y*ratio[1]))
            for z in range(targetShape[2]):
                zDisc = int(math.floor(z*ratio[2]))
                # sample the center
                if opt == 0:
                    threeDMatRedux[x,y,z] = data[xDisc, yDisc, zDisc]
                else:
                    box = data[max(0,xDisc-lowBoxExtension[0]):min(BRAIN_SZ[0],xDisc+highBoxExtension[0]),
                                max(0,yDisc-lowBoxExtension[1]):min(BRAIN_SZ[1],yDisc+highBoxExtension[1]),
                                max(0,zDisc-lowBoxExtension[2]):min(BRAIN_SZ[2],zDisc+highBoxExtension[2])]

                    if not box.any():
                        continue

                    # average pooling
                    if opt == 1:
                        threeDMatRedux[x,y,z] = np.mean(box)
                    # random pooling
                    elif opt == 2:
                        threeDMatRedux[x,y,z] = np.random.choice(np.asarray(box).reshape(-1))
                    # max sampling
                    elif opt == 3:
                        threeDMatRedux[x,y,z] = np.max(box)

    # write out to a file if filename is specified
    if filename:
        np.save(filename, threeDMatRedux)

    return threeDMatRedux

def loadfALFF_All():
    """
    Dumps pooled data under ./pooledData directory
    Make sure you already have ./pooledData created
    """
    for i in xrange(1,1072):
        if i%25==0:
            print i
        data = loadfALFF(i)
        np.save('pooledData/original_'+str(i), data)
        sizeReduction(data, (45, 54, 45), opt=1, poolBox=(2,2,2), filename='pooledData/avgPool_'+str(i)+'_reduce2')
        sizeReduction(data, (30, 36, 30), opt=1, poolBox=(3,3,3), filename='pooledData/avgPool_'+str(i)+'_reduce3')
        sizeReduction(data, (45, 54, 45), opt=2, poolBox=(2,2,2), filename='pooledData/randomPool_'+str(i)+'_reduce2')
        sizeReduction(data, (30, 36, 30), opt=2, poolBox=(3,3,3), filename='pooledData/randomPool_'+str(i)+'_reduce3')
        sizeReduction(data, (45, 54, 45), opt=3, poolBox=(2,2,2), filename='pooledData/maxPool_'+str(i)+'_reduce2')
        sizeReduction(data, (30, 36, 30), opt=3, poolBox=(3,3,3), filename='pooledData/maxPool_'+str(i)+'_reduce3')

BRAIN_REGIONS = 116
def loadROI(patientID):
    """
    Converts the fALFF ROI data to covariance matricies.

    @type   patientID   :   int
    @param  patientID   :   id of the patient to load the brain.

    @rtype              :   2D numpy matrix
    @return             :   Covariance matrix
    """
    with open('/data/processed_phenotype_data.csv') as csvfile:
        id = next(itertools.islice(csv.reader(csvfile), patientID, None))[2]

    count = 0
    fileName = '/data/CS_273B_Final_Project/'+str(id)+' _data.csv'
    if not os.path.isfile(fileName):
        return

    csvfile = open(fileName, 'rb')
    csvR = csv.reader(csvfile)
    row_count = sum(1 for row in csvR)-1
    csvfile.seek(0)

    roiMat = np.empty((row_count,BRAIN_REGIONS),dtype='float')
    first = True
    count = 0
    for row in csvR:
        if first:
            first = False
            continue

        roiMat[count,:] = [float(s) for s in row[1:]]
        count += 1

    return np.cov(np.transpose(roiMat))

def loadROI_All():
    """
    Dumps covariance data under ./ROI directory
    Make sure you already have ./ROI created
    """
    for i in xrange(1,1072):
        covData = loadROI(i)
        if covData is not None:
            np.save('./ROI/'+str(i), covData)

import h5py
def write2hdf5(filename, dict2store):
    """
    Write items in a dictionary to an hdf5file

    @type   filename    :   String
    @param  filename    :   Filename of the hdf5 file to output to.
    @type   dict2store  :   Dict
    @param  dict2store  :   Dictionary of items to store. The value should be an array.

    """
    with h5py.File(filename,'w') as hf:
        for key,value in dict2store.iteritems():
            hf.create_dataset(key, data=value,compression="lzf")

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
    import tensorflow as tf
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

def blackOutBrain(brainMat, brainRegions):
    """
    'Blacks out' the brain regions specified with 'brainRegions'

    @type   brainMat    :   3D numpy matrix
    @param  brainMat    :   describes the brain
    @type   brainRegion :   int array
    @param  brainRegion :   Regions of the brain regions to black out.
                            The range should be [1,116], so 1 Indexed!

    @rtype  brainMatRet :   3D numpy matrix
    @return brainMatRet :   Describes a brain with specified regions blacked out
    """
    brainMatRet = np.copy(brainMat)

    # find the id of the brainRegion
    brainIDs = brainRegion2brainID(brainRegions)

    # figure out which voxels blong to the brain regions
    blackOutVoxels = [0]
    with open('/data/region_code.csv') as csvfile:
        csvR = csv.reader(csvfile)
        next(csvR)
        for row in csvR:
            if int(row[1]) in brainIDs:
                blackOutVoxels.append(int(row[0]))

    # figure out where the blackout voxels are located
    with open('/data/coord.csv') as csvfile:
        csvR = csv.reader(csvfile)
        next(csvR)
        for i in range(len(blackOutVoxels)-1):
            for j in range(blackOutVoxels[i+1]-blackOutVoxels[i]-1):
                next(csvR)
            coord = [int(i) for i in next(csvR)]
            brainMatRet[coord[1]-1, coord[2]-1, coord[3]-1] = 0

    return brainMatRet

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

    with open('/data/region_code.csv') as csvfile:
        csvR = csv.reader(csvfile)
        next(csvR)

        count = 0
        for row in csvR:
            coord = coords[count,:]
            brainMat[coord[0],coord[1],coord[2]] = weights[id2regionDict[int(row[1])]]
            count += 1

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
            plt.imshow(mat[:,:,zLocs[i]], cmap = "gray", interpolation='none')
        else:
            plt.imshow(mat[:,:,zLocs[i]], vmin=min(valRange), vmax=max(valRange), cmap = "gray", interpolation='none')

    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cax=cax)

    plt.savefig(filename)

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