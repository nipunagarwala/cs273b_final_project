# README for utils_fMRI_augment.py

## Global Variables
  * `ALL_BRAINS` - list containing 3D numpy brain matrix for every patient. Generated with prepreProcess()
  * `BRAINID2COORDS` - Brain ID to corresponding coordinates. Generated with prepreProcess()

## Data Generation
  * `loadfALFF(patientID)` - Converts raw fALFF data to a 3D representation in numpy.
		** `loadfALFF_All()` - Wraps around loadfALFF(), dumps pooled data under ./pooledData directory
  * `sizeReduction(data, targetShape, opt, poolBox=(2,2,2), filename=None)` - Reduce size of 3D brain data
  * `augmentROI2Brain(patientID)` - Generates the ROI brain data and stores it

## Data Augmentation
  * `augmentGeoTrans(patientID)` - Applies geometric transformations (reflection in 3 dimensions), and saves to a file
  * `augmentCompleteSwapWorker(num)` - Creates and saves complete swapout brain (autistic and control) with the name modified with 'num'
  * `augmentPartialSwap(numStealRegions)` - Multiprocess partial brain swap function
    * `augmentPartialSwapWorker(runList)` - Creates and saves partial swapout brain
  * `augmentBlackout(runList)` - Multiprocess brain blackout function
    * `augmentBlackoutWorker(runList)` - Creates and saves blackout brain
  * `batchROIData(patientID)` - Batches up the ROI data

## Helper Functions
  * `getGroupLabels()` - Returns 2 lists, each listing autistic or control patient IDs
  * `blackOutBrain(brainMat, brainRegions)` - 'Blacks out' the brain regions specified with 'brainRegions'
  * `augmentPatchwork(patientID, numStealRegions, autistic, blackout)` - Swapout/blackout data augmentation
  * `getBrainRegion(patientID, brainRegions)` - Gets the brain regions specified with 'brainRegions' and adds to partialBrain
  * `prepreProcess()` - Loads every brain npy into ALL_BRAINS and BRAINID2COORDS
  * `returnFeatures(patientID)` - Returns patient label (0 for control, 1 for autistic) and phenotype data
  * `convertBrain2bin(patientID, brain_data, bin_path)` - Combines 3D brain info in brain_data with phenotype data, and stores it under bin_path
  * `executeAugFunc(func, mapList)` - Multiprocess a function

## Others
  * `reduceMaxAvg()` - Takes pooled data (average, max) from /data/pooledData directory, appends phenotype data, stores the output binary file, and records the json file
  * `reduceHDF5Sz(patientID)` - hdf5 zips the roi data
  * `generateJSONTrain(dataDir, outputName)`

## JSON Related
  * `scrapeTestAndTrain(jsonAll, jsonTest, jsonTrain, regexStr, outPrefix='')` - Generate npy including patient IDs for all, test, train (specified via the input json files)
  * `generateJSON(dataDir, regexList, outputName, testIDs)`
  * `combineROITestJSON(filename, outPath, batchSz, reduction)`
