from operator import truediv
from utils_visual import *
from utils import *
import os

import numpy as np
import csv
import itertools
import math

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

import re
def augmentGeoTrans(originalDir='/data/originalfALFFData', outDir = '/data/geoTrans'):
    for file in os.listdir(originalDir):
        filepath = os.path.join(originalDir,file)
        brain = np.load(filepath)

        patientID = re.search('[0-9]+',file).group(0)

        for i in range(8):
            outfilepath = os.path.join(outDir,str(patientID)+'_geoTrans_'+str(i))
            flipped= brain.copy()
            if i&1:
                flipped = brain[::-1,::-1,:]
            np.save(flipped, outfilepath)

augmentGeoTrans()


def augment():
    pass