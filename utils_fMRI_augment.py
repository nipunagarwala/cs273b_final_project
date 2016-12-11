from operator import truediv
from utils_visual import *
from utils import *
from multiprocessing import Pool

import os
import random
import numpy as np
import csv
import itertools
import math
import pickle

BRAIN_SZ = (91,109,91)
BRAIN_REGION_SZ = 116

BRAIN_DIR = '/data/originalfALFFData'

ALL_BRAINS = None
BRAINID2COORDS = None

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

def reduceHDF5Sz(patientID):
    with open('/data/processed_phenotype_data.csv') as csvfile:
        idd = next(itertools.islice(csv.reader(csvfile), patientID, None))[2]

    fileName = '/data/CS_273B_Final_Project/'+str(idd)+' _data.csv'
    if not os.path.isfile(fileName):
        return

    outputPath = '/data/augmented_roi_pooled_%d/%d_roi_pooled_%d.hdf5'
    roiPath = '/data/augmented_roi_original/%d_roi.hdf5'
    reduxs = [4,5,6]
    if os.path.isfile(outputPath % (reduxs[0],patientID,reduxs[0])):
        return

    h5Dict = hdf52dict(roiPath%patientID)

    newH5Dict = {}
    for redux in reduxs:
        newH5Dict[redux] = {}

    for key in h5Dict.keys():
        k = key
        brain = h5Dict[key]
        for redux in reduxs:
            newSz = [int(round(float(i)/redux)) for i in BRAIN_SZ]
            newH5Dict[redux][key] = sizeReduction(brain, newSz, opt=1, poolBox=(redux,redux,redux))

    for redux in reduxs:
        write2hdf5(outputPath % (redux,patientID,redux), newH5Dict[redux], compression='lzf')


def augmentGeoTrans(patientID, originalDir='/data/originalfALFFData', outDir='/data/augmented_geoTrans'):
    filepath = os.path.join(originalDir,'original_%d.npy'%patientID)
    brain = np.load(filepath)

    for i in range(8):
        outfilepath = os.path.join(outDir,str(patientID)+'_geoTrans_'+str(i))
        flipped= brain.copy()
        if i&1:
            flipped = flipped[::-1,:,:]
        if i&2:
            flipped = flipped[:,::-1,:]
        if i&4:
            flipped = flipped[:,:,::-1]
        np.save(outfilepath, flipped)

def getGroupLabels(filename='/data/processed_phenotype_data.csv'):
    autismIDs = []
    controlIDs = []
    with open(filename) as csvfile:
        csvR = csv.reader(csvfile)
        next(csvR)
        for row in csvR:
            if row[3]=='0':
                controlIDs.append(int(row[0])) 
            else:
                autismIDs.append(int(row[0])) 
    return sorted(autismIDs),sorted(controlIDs)

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
    global ALL_BRAINS,BRAINID2COORDS
    brainMatRet = np.copy(brainMat)

    for region in brainRegions:
        for x,y,z in BRAINID2COORDS[region]:
            brainMatRet[x,y,z] = 0

    return brainMatRet

def getBrainRegion(patientID, brainRegions, partialBrain=None):
    """

    @type   brainRegion :   int array
    @param  brainRegion :   Regions of the brain regions to black out.
                            The range should be [1,116], so 1 Indexed!

    @rtype  brainMatRet :   3D numpy matrix
    @return brainMatRet :   
    """
    # if type(partialBrain):
    #     partialBrain = np.zeros(BRAIN_SZ)
    global ALL_BRAINS,BRAINID2COORDS

    if ALL_BRAINS==None:
        # load the base brain
        filepath = os.path.join(BRAIN_DIR,'original_%d.npy'%patientID)
        brain = np.load(filepath)
    else:
        brain = ALL_BRAINS[patientID-1]

    for region in brainRegions:
        for x,y,z in BRAINID2COORDS[region]:
            partialBrain[x,y,z] = brain[x,y,z]

    return partialBrain

def augmentPatchwork(patientID=None, numStealRegions=None, autistic=None, blackout=False):
    global ALL_BRAINS,BRAINID2COORDS
    autismIDs,controlIDs = getGroupLabels()

    # only augment the training set!
    testIDs = np.load('/data/testPatientIDs.npy').tolist()
    if patientID!=None and patientID in testIDs:
        return None

    if patientID:
        # base the patchwork from a brain specified
        if ALL_BRAINS==None:
            # load the base brain
            filepath = os.path.join(BRAIN_DIR,'original_%d.npy'%patientID)
            brain = np.load(filepath)
        else:
            brain = ALL_BRAINS[patientID-1]

        ids = autismIDs if (patientID in autismIDs) else controlIDs
        ids.remove(patientID)

        # now steal brain regions from other brains
        regions2steal = random.sample(list(xrange(1,BRAIN_REGION_SZ+1)),numStealRegions)

        # black out the base brain
        brain = blackOutBrain(brain, regions2steal)

        if blackout:
            return brain
    else:
        # completely make a patchwork from a scratch
        if autistic==None:
            autistic = random.choice([True, False])
        ids = autismIDs if autistic else controlIDs
        brain = np.zeros(BRAIN_SZ)
        regions2steal = list(xrange(1,BRAIN_REGION_SZ+1))

    # remove testing samples from the ids
    for i in testIDs:
        if i in ids:
            ids.remove(i)

    # add the brain regions
    for region in regions2steal:
        getBrainRegion(random.choice(ids), [region], partialBrain=brain)

    return brain

def augmentPatchworkWorker(num):
    if not os.path.isfile('/data/augmented_swap_all/'+str(num)+'_autism_all_patched.npy'):
        autistic = augmentPatchwork(autistic=True)
        if autistic!=None:
            np.save('/data/augmented_swap_all/'+str(num)+'_autism_all_patched',autistic)
    if not os.path.isfile('/data/augmented_swap_all/'+str(num)+'_control_all_patched.npy'):
        control = augmentPatchwork(autistic=False)
        if control!=None:
            np.save('/data/augmented_swap_all/'+str(num)+'_control_all_patched',control)

def augmentPatchworkPartialWorker(runList):
    num = runList[0]
    filename = runList[1]
    if not os.path.isfile(filename+'.npy'):
        brain = augmentPatchwork(patientID=num, numStealRegions=25)
        if brain!=None:
            np.save(filename,brain)

def augmentPatchworkPartial():
    autistic,control = getGroupLabels()

    runList = []
    for i in range(5):
        for a in autistic:
            runList.append((a,'/data/augmented_swap_partial/'
                              +str(a)+'_autism_partially_patched_'+str(i)))
        for c in control:
            runList.append((c,'/data/augmented_swap_partial/'
                              +str(c)+'_control_partially_patched_'+str(i)))

    p = Pool(8)
    p.map(augmentPatchworkPartialWorker,runList)

def augmentBlackoutWorker(runList):
    num = runList[0]
    filename = runList[1]
    if not os.path.isfile(filename+'.npy'):
        brain = augmentPatchwork(patientID=num, numStealRegions=25, blackout=True)
        if brain!=None:
            np.save(filename,brain)

def augmentBlackout():
    autistic,control = getGroupLabels()

    runList = []
    for i in range(5):
        for a in autistic:
            runList.append((a,'/data/augmented_blackout/'
                              +str(a)+'_autism_blackout_'+str(i)))
        for c in control:
            runList.append((c,'/data/augmented_blackout/'
                              +str(c)+'_control_blackout_'+str(i)))

    p = Pool(8)
    p.map(augmentBlackoutWorker,runList)

def augmentROI2Brain(patientID):
    outpath = '/data/augmented_roi_original'
    with open('/data/processed_phenotype_data.csv') as csvfile:
        id = next(itertools.islice(csv.reader(csvfile), patientID, None))[2]

    fileName = '/data/CS_273B_Final_Project/'+str(id)+' _data.csv'
    if not os.path.isfile(fileName) or os.path.isfile(outpath+'/%d_roi.hdf5'%patientID):
        return

    print patientID

    csvfile = open(fileName, 'rb')
    csvR = csv.reader(csvfile)
    next(csvR)

    roiDict = {}
    for i,row in enumerate(csvR):
        roiDict['step_%d'%i] = weights2Brain([float(s) for s in row[1:]])

    write2hdf5(os.path.join(outpath,'%d_roi.hdf5'%patientID), roiDict, compression='lzf')

def compressROI(patientID):
    datapath = '/data/augmented_roi_original'

    with open('/data/processed_phenotype_data.csv') as csvfile:
        pat_id = next(itertools.islice(csv.reader(csvfile), patientID, None))[2]

    count = 0
    fileName = '/data/CS_273B_Final_Project/%s _data.csv' %pat_id
    if not os.path.isfile(fileName):
        return

    csvfile = open(fileName, 'rb')
    csvR = csv.reader(csvfile)
    row_count = sum(1 for row in csvR)-1
    csvfile.seek(0)

    roiDict = {}
    for i in range(row_count):
        npyFile = os.path.join(datapath,'%d_roi_step_%d.npy'%(patientID,i))
        if not os.path.isfile(npyFile):
            print 'fail'
            return
        roiDict['step_%d'%i] = np.load(npyFile)

    write2hdf5(os.path.join(datapath,'%d_roi.hdf5'%patientID), roiDict, compression='lzf')

def autismVScontrol():
    autistic,control = getGroupLabels()
    a1 = random.choice(autistic)
    a2 = random.choice(autistic)
    c1 = random.choice(control)
    c2 = random.choice(control)

    # load the base brain
    brainA1 = np.load(os.path.join(BRAIN_DIR,'original_%d.npy'%a1))
    brainA2 = np.load(os.path.join(BRAIN_DIR,'original_%d.npy'%a2))
    brainC1 = np.load(os.path.join(BRAIN_DIR,'original_%d.npy'%c1))
    brainC2 = np.load(os.path.join(BRAIN_DIR,'original_%d.npy'%c2))

    ran = 0.3
    mat2visual(brainA1-brainA2,[20,30,40,50,60],'autism_autism.png',[-ran,ran])
    mat2visual(brainC1-brainC2,[20,30,40,50,60],'control_control.png',[-ran,ran])
    mat2visual(brainA1-brainC1,[20,30,40,50,60],'control_autism.png',[-ran,ran])
    mat2visual(brainA1-brainC2,[20,30,40,50,60],'control_autism2.png',[-ran,ran])
    mat2visual(brainA2-brainC1,[20,30,40,50,60],'control_autism3.png',[-ran,ran])
    mat2visual(brainA2-brainC2,[20,30,40,50,60],'control_autism4.png',[-ran,ran])

    mat2visual(brainA1,[20,30,40,50,60],'autism1.png')
    mat2visual(brainA2,[20,30,40,50,60],'autism2.png')
    mat2visual(brainC1,[20,30,40,50,60],'control1.png')
    mat2visual(brainC2,[20,30,40,50,60],'control2.png')
    
def scrapeTestAndTrain():
    import json
    import re
    
    with open(os.path.join('/data','reduced_all2.json')) as data_file:    
        data = json.load(data_file)

        allFilenames = [re.search('[0-9]+',re.search('[0-9]+.bin', filename).group(0)).group(0) for filename in data]
        allPatient = sorted([int(i) for i in allFilenames])

    with open(os.path.join('/data','reduced_test2.json')) as data_file:    
        data = json.load(data_file)

        testFilenames = [re.search('[0-9]+',re.search('[0-9]+.bin', filename).group(0)).group(0) for filename in data]
        testPatient = sorted([int(i) for i in testFilenames])

    with open(os.path.join('/data','reduced_train2.json')) as data_file:    
        data = json.load(data_file)

        trainFilenames = [re.search('[0-9]+',re.search('[0-9]+.bin', filename).group(0)).group(0) for filename in data]
        trainPatient = sorted([int(i) for i in trainFilenames])

    np.save('/data/allPatientIDs', np.asarray(allPatient))
    np.save('/data/testPatientIDs', np.asarray(testPatient))
    np.save('/data/trainPatientIDs', np.asarray(trainPatient))

def prepreProcess():
    global ALL_BRAINS,BRAINID2COORDS
    ALL_BRAINS = []
    for patientID in xrange(1,1072):
        # load the base brain
        filepath = os.path.join(BRAIN_DIR,'original_%d.npy'%patientID)
        ALL_BRAINS.append(np.load(filepath))

    BRAINID2COORDS = pickle.load(open('/data/brainRegionID2Coords.p','rb'))

    print 'done with preprocessing...'

if __name__ == '__main__':
    # prepreProcess()
    # func = augmentPatchworkWorker
    # mapList = list(range(5000))

    # p = Pool(8)
    # p.map(func, mapList)

    # augmentPatchworkPartial()
    # augmentBlackout()

    #executeAugFunc(reduceNpySz,os.listdir('/data/augmented_roi_original'))
    #executeAugFunc(augmentROI2Brain,xrange(1,1072))

    #a = augmentPatchwork(patientID=2, numStealRegions=25, autistic=None)

    # dird = '/data/augmented_swap_all/'
    # # dird = '/data/augmented_swap_partial/'
    # ab = os.listdir(dird)
    # for i in range(1):
    #     filename = os.path.join(dird, random.choice(ab))
    #     a = np.load(filename)
    #     mat2visual(a,[20,40,60],'swap_all%d.png'%i)
    pass