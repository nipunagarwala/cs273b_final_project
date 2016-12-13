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

def reduceMaxAvg():
    import json
    from create_brain_binaries import _normalize_brain,_create_feature_binary

    testIDs = np.load('/data/useful_npy/testPatientIDs.npy').tolist()

    inputDir = '/data/pooledData/'
    outputDir = '/data/binaries_%s_reduce_3/'
    poolMethods = ['avgPool','maxPool']

    testFiles = {'avgPool':[],'maxPool':[]}
    trainFiles = {'avgPool':[],'maxPool':[]}
    for patientID in xrange(1,1072):
        print patientID
        for poolMethod in poolMethods:
            brainMat = np.load(os.path.join(inputDir,'%s_%d_reduce3.npy'%(poolMethod,patientID)))
            brainMat = brainMat.astype(np.float32)
            normailized_batchedMat = _normalize_brain(brainMat)

            patient_label,phenotype_data = returnFeatures(patientID)

            bin_path = os.path.join(outputDir%poolMethod, '%s_%d_reduce3.bin' % (poolMethod,patientID))
            _create_feature_binary(phenotype_data, normailized_batchedMat, patient_label, bin_path)

            if patientID in testIDs:
                testFiles[poolMethod].append(bin_path)
            else:
                trainFiles[poolMethod].append(bin_path)

    json.dump(testFiles['avgPool'], open('/data/test_reduce3_avgPool.json','w'))
    json.dump(trainFiles['avgPool'], open('/data/train_reduce3_avgPool.json','w'))
    json.dump(testFiles['maxPool'], open('/data/test_reduce3_maxPool.json','w'))
    json.dump(trainFiles['maxPool'], open('/data/train_reduce3_maxPool.json','w'))

def reduceHDF5Sz(patientID):
    with open('/data/processed_phenotype_data.csv') as csvfile:
        idd = next(itertools.islice(csv.reader(csvfile), patientID, None))[2]

    fileName = '/data/CS_273B_Final_Project/'+str(idd)+' _data.csv'
    if not os.path.isfile(fileName):
        return

    outputPath = '/data/augmented_roi_pooled_%d/%d_roi_pooled_%d.hdf5'
    roiPath = '/data/augmented_roi_original/%d_roi.hdf5'
    reduxs = [4,5,6]
    if os.path.isfile(outputPath % (reduxs[1],patientID,reduxs[1])):
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
    testIDs = np.load('/data/useful_npy/testPatientIDs.npy').tolist()
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
    numStealRegions = runList[2]
    if not os.path.isfile(filename+'.npy'):
        brain = augmentPatchwork(patientID=num, numStealRegions=numStealRegions)
        if brain!=None:
            np.save(filename,brain)

def augmentPatchworkPartial(numStealRegions):
    autistic,control = getGroupLabels()

    runList = []
    for i in range(10):
        for a in autistic:
            runList.append((a,'/data/augmented_swap_partial_steal_%d/%d_autism_partially_patched_%d' 
                               % (numStealRegions,a,i),numStealRegions))
        for c in control:
            runList.append((c,'/data/augmented_swap_partial_steal_%d/%d_control_partially_patched_%d' 
                               % (numStealRegions,c,i),numStealRegions))

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
    
def scrapeTestAndTrain(jsonAll, jsonTest, jsonTrain, regexStr, outPrefix=''):
    import json
    import re
    
    with open(os.path.join('/data',jsonAll)) as data_file:    
        data = json.load(data_file)

        allFilenames = [re.search('[0-9]+',re.search(regexStr, filename).group(0)).group(0) for filename in data]
        allPatient = sorted([int(i) for i in allFilenames])

    with open(os.path.join('/data',jsonTest)) as data_file:    
        data = json.load(data_file)

        testFilenames = [re.search('[0-9]+',re.search(regexStr, filename).group(0)).group(0) for filename in data]
        testPatient = sorted([int(i) for i in testFilenames])

    with open(os.path.join('/data',jsonTrain)) as data_file:    
        data = json.load(data_file)

        trainFilenames = [re.search('[0-9]+',re.search(regexStr, filename).group(0)).group(0) for filename in data]
        trainPatient = sorted([int(i) for i in trainFilenames])

    np.save('/data/useful_npy/%sallPatientIDs'%outPrefix, np.asarray(list(set(allPatient))))
    np.save('/data/useful_npy/%stestPatientIDs'%outPrefix, np.asarray(list(set(testPatient))))
    np.save('/data/useful_npy/%strainPatientIDs'%outPrefix, np.asarray(list(set(trainPatient))))

def prepreProcess():
    global ALL_BRAINS,BRAINID2COORDS
    ALL_BRAINS = []
    for patientID in xrange(1,1072):
        # load the base brain
        filepath = os.path.join(BRAIN_DIR,'original_%d.npy'%patientID)
        ALL_BRAINS.append(np.load(filepath))

    print 'done with preprocessing...'

PHENOTYPE_FILE = os.path.abspath('/data/normalized_imputed_phenotype_data.csv')
def returnFeatures(patientID, phenotype_file=PHENOTYPE_FILE):
    with open(phenotype_file, 'r') as csvfile:
        patient_reader = csv.reader(csvfile)
        for i in range(patientID):
            next(patient_reader)

        # Retrieve data from phenotype CSV
        patient = next(patient_reader)
        patient_label = np.float32(patient[3])
        phenotype_data = patient[5:16] + patient[19:]
        phenotype_data = np.asarray(phenotype_data, dtype=np.float32)

    return patient_label,phenotype_data

def convertBrain2npy(patientID, brain_data, bin_path):
    from create_brain_binaries import _normalize_brain,_create_feature_binary

    patient_label,phenotype_data = returnFeatures(patientID)

    # Load brain images from .npy files
    brain_data = brain_data.astype(np.float32)
    normailized_brain = _normalize_brain(brain_data)

    # Create binaries from all data
    _create_feature_binary(phenotype_data, normailized_brain, patient_label, bin_path)

def batchROIData(patientID, batchSz=10):
    import re
    from create_brain_binaries import _normalize_brain,_create_feature_binary

    with open('/data/processed_phenotype_data.csv') as csvfile:
        idd = next(itertools.islice(csv.reader(csvfile), patientID, None))[2]

    # check if ROI file exists for this patient
    fileName = '/data/CS_273B_Final_Project/'+str(idd)+' _data.csv'
    if not os.path.isfile(fileName):
        return

    inputPath = '/data/augmented_roi_pooled_%d/%d_roi_pooled_%d.hdf5'
    outputPath = '/data/augmented_roi_pooled_%d_batched/%d_roi_pooled_%d_batched.hdf5'
    reduxs = [4,5,6]
    if os.path.isfile(outputPath % (reduxs[1],patientID,reduxs[1])):
        return

    patient_label,phenotype_data = returnFeatures(patientID)

    for redux in reduxs:
        h5Dict = hdf52dict(inputPath % (redux,patientID,redux))

        maxStep = max([int(re.search('[0-9]+',i).group(0)) for i in h5Dict.keys()])
        npyShape = h5Dict['step_0'].shape
        allSteps = np.zeros((npyShape[0], npyShape[1], npyShape[2], maxStep))
        for i in range(maxStep):
            allSteps[:,:,:,i] = h5Dict['step_%d'%i]

        for i in range(maxStep-batchSz):
            batchedMat = allSteps[:,:,:,i:i+batchSz].copy()

            batchedMat = batchedMat.astype(np.float32)
            normailized_batchedMat = _normalize_brain(batchedMat)

            # Create binaries from all data
            bin_path = '%d_roi_batchSz_%d_index_%d_reduction_%d.bin' % (patientID,batchSz,i,redux)
            outDir = '/data/binaries_roi_batchSz_%d_reduced_%d' % (batchSz,redux)
            _create_feature_binary(phenotype_data, normailized_batchedMat, 
                                   patient_label, os.path.join(outDir, bin_path))

def generateJSON(dataDir, regexList, outputName, testIDs):
    import re
    import json

    filenames = os.listdir(dataDir)

    testList = []
    trainList = []
    for filename in filenames:
        regStr = filename
        for rex in regexList:
            regStr = re.search(rex,regStr).group(0)
        patientID = int(regStr)

        if patientID in testIDs:
            testList.append(os.path.join(dataDir,filename))
        else:
            trainList.append(os.path.join(dataDir,filename))

    json.dump(testList,open('/data/test_'+outputName+'.json','w'))
    json.dump(trainList,open('/data/train_'+outputName+'.json','w'))


def generateJSONTrain(dataDir, outputName):
    import json

    trainList = os.listdir(dataDir)
    json.dump(trainList,open('/data/train_'+outputName+'.json','w'))

def combineROITestJSON(filename, outPath, batchSz, reduction):
    import json
    import re

    files = json.load(open(filename,'r'))

    patientFiles = {}
    for file in files:
        patientID = int(re.search('[0-9]+',re.search('[0-9]+_roi_batchSz',file).group(0)).group(0))
        if patientID not in patientFiles:
            patientFiles[patientID] = []
        patientFiles[patientID].append(file)

    for k in patientFiles.keys():
        json.dump(patientFiles[k],open(os.path.join(outPath,'roi_batchSz_%d_reduction_%d_test_patientID_%d.json'
                                       %(batchSz,reduction,k)),'w'))

def split_list(idList, split_fraction):
    # Calculate number of files to split
    num_files = len(idList)
    num_train = int(num_files * split_fraction)

    # Permute files
    perm = np.arange(num_files)
    np.random.shuffle(perm)
    idList = [idList[i] for i in perm]

    # Split file list
    train_files = idList[:num_train]
    test_files = idList[num_train:]
    return train_files,test_files

def createBlackRegions(patientID):
    autistic,control = getGroupLabels()
    patLabel = 'autistic' if patientID in autistic else 'control'

     # load the base brain
    filepath = os.path.join(BRAIN_DIR,'original_%d.npy'%patientID)
    brain = np.load(filepath)

    for i in xrange(1,BRAIN_REGION_SZ+1):
        blackBrain = blackOutBrain(brain, [i])
        np.save('/data/blackoutBrains/%d_%s_region_%d'%(patientID,patLabel,i), blackBrain)

if __name__ == '__main__':
    for i in xrange(10,21):
        print np.memmap(filename='/data/swap_partial_binaries_reduced/compressed_%d_autism_partially_patched_4.bin'%i, dtype='float32',mode='r',shape=1)
    # BRAINID2COORDS = pickle.load(open('/data/useful_npy/brainRegionID2Coords.p','rb'))

    # prepreProcess()
    # augmentPatchworkPartial(25)

    # func = batchROIData
    # mapList = list(xrange(1,1072))

    # p = Pool(8)
    # p.map(func, mapList)

    # augmentPatchworkPartial(25)
    # reduceMaxAvg()

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

    # mat2visual(ALL_BRAINS[235],[20,40,60],'allbrain.png')

    # a = np.load('/data/originalfALFFData/original_236.npy')
    # mat2visual(a,[20,40,60],'npyBrain.png')

    
    # a = augmentPatchwork(patientID=27, numStealRegions=25)
    # if a!=None:
    #     mat2visual(a,[20,40,60],'a.png')
    # else:
    #     print 'None'

    # generateJSON('/data/binaries_avgPool_reduce_3', ['avgPool_[0-9]+','[0-9]+'], 'avgPool_reduce_3', np.load('/data/useful_npy/testPatientIDs.npy').tolist())
    # generateJSON('/data/binaries_maxPool_reduce_3', ['maxPool_[0-9]+','[0-9]+'], 'maxPool_reduce_3', np.load('/data/useful_npy/testPatientIDs.npy').tolist())


    # generateJSON('/data/binaries_roi_batchSz_10_reduced_4', ['[0-9]+_roi','[0-9]+'], 'roi_batchSz_10_reduce_4', np.load('/data/useful_npy/roi_testPatientIDs.npy').tolist())
    # generateJSON('/data/binaries_roi_batchSz_10_reduced_5', ['[0-9]+_roi','[0-9]+'], 'roi_batchSz_10_reduce_5', np.load('/data/useful_npy/roi_testPatientIDs.npy').tolist())
    # generateJSON('/data/binaries_roi_batchSz_10_reduced_6', ['[0-9]+_roi','[0-9]+'], 'roi_batchSz_10_reduce_6', np.load('/data/useful_npy/roi_testPatientIDs.npy').tolist())

    # combineROITestJSON('/data/test_roi_batchSz_10_reduce_4.json', '/data/roi_batchSz_10_reduction_4_test_json', 10, 4)
    # combineROITestJSON('/data/test_roi_batchSz_10_reduce_5.json', '/data/roi_batchSz_10_reduction_5_test_json', 10, 5)
    # combineROITestJSON('/data/test_roi_batchSz_10_reduce_6.json', '/data/roi_batchSz_10_reduction_6_test_json', 10, 6)

    # import json
    # import re

    # testIDs = np.load('/data/useful_npy/testIDs_20_80_split.npy').tolist()
    # filenames = os.listdir('/data/binaries_reduced2')
    # testFilenames = []
    # for f in filenames:
    #     if int(re.search('[0-9]+',re.search('compressed_[0-9]+',f).group(0)).group()) in testIDs:
    #         testFilenames.append('/data/binaries_reduced2/'+f)
    
    # json.dump(testFilenames,open('/data/test_20_80_split.json','w'))