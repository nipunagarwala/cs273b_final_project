from utils_visual import *
from operator import truediv

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
#import tensorflow as tf
from sklearn.metrics import confusion_matrix


from mpl_toolkits.axes_grid1 import make_axes_locatable

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

def bin2npy(dirName='/data/binaries_reduced'):
    LABEL_SZ = 1
    PHENO_SZ = 29
    X_SZ = 31
    Y_SZ = 37
    Z_SZ = 31
    for filename in os.listdir(dirName):
        print os.path.join(dirName,filename)
        brain = np.memmap(filename=os.path.join(dirName,filename), dtype='float32',
                          mode='r', offset=(LABEL_SZ+PHENO_SZ)*4, shape=(X_SZ,Y_SZ,Z_SZ))
        np.save(os.path.join('/data/binaries_reduced_npy/',filename), brain)

def extract_parser():
    parser = argparse.ArgumentParser(description='Evaluation procedure for Salami CNN.')
    network_group = parser.add_mutually_exclusive_group()
    data_group = parser.add_mutually_exclusive_group()
    checkoint_file_group = parser.add_mutually_exclusive_group()
    data_group.add_argument('--train', action="store_true", help='Training the model')
    data_group.add_argument('--test', action="store_true", help='Testing the model')
    data_group.add_argument('--blackout', action="store_true", help='Conduct Blackout visualization')
    network_group.add_argument('--model', choices=['ae', 'cae', 'cnn', 'nn', 'mmnn'],
                        default='mmnn', help='Select model to run.')
    parser.add_argument('--chkPointDir', dest='chkPt', default='/data/axel_ckpt/cnn_swap_partial_NOT_WORKING',
                        help='Directory to save the checkpoints. Default is /data/ckpt')
    parser.add_argument('--numIters', dest='numIters', default=200, type=int,
                        help='Number of Training Iterations. Default is 200')
    parser.add_argument('--overrideChkpt', dest='overrideChkpt', action="store_true",
                        help='Override the checkpoints')

    parser.add_argument('--dataDir', dest='dataDir', default='',
                        help='Directory to load the samples from.')

    parser.set_defaults(overrideChkpt=False)
    return parser.parse_args()


import zipfile
import shutil
from multiprocessing import Pool

def zipper(dataPack):
    # a helper function for zipDirectory()
    fromDir, filenames, toDir = dataPack

    with zipfile.ZipFile(toDir, 'w', zipfile.ZIP_DEFLATED) as f:
        for filename in filenames:
            fullname = os.path.join(fromDir, filename)
            f.write(fullname, arcname=filename)

def zipDirectory(dataDir, outputDirName=None, zipSz=250):
    # compress the directories into chunks of zip files
    poolWorkerCount = 8

    if dataDir[-1]=='/':
        dataDir = dataDir[:-1]

    if outputDirName==None:
        outputDirName = '%s_compressed' % dataDir

    print "Zipping up %s to %s" %(dataDir,outputDirName)
    if not os.path.exists(outputDirName):
        os.mkdir(outputDirName)

    filenames = os.listdir(dataDir)
    zipCount = int((zipSz<<20)/os.stat(os.path.join(dataDir, filenames[0])).st_size)
    if len(filenames) < poolWorkerCount*zipCount:
        zipCount = len(filenames)/poolWorkerCount + 1

    zipNum = 0
    count = 0
    zipNames = []
    dataPacks = []
    for filename in filenames:
        count += 1
        zipNames.append(filename)

        if count == zipCount:
            toDir = '%s/compressed_%d' % (outputDirName, zipNum)
            dataPacks.append((dataDir, zipNames, toDir))

            zipNum += 1
            count = 0
            zipNames = []

    if count != 0:
        toDir = '%s/compressed_%d' % (outputDirName, zipNum)
        dataPacks.append((dataDir, zipNames, toDir))

    p = Pool(poolWorkerCount)
    p.map(zipper, dataPacks)

def unzipper(dataPack):
    fullname, testDir = dataPack
    zip_ref = zipfile.ZipFile(fullname, 'r')
    zip_ref.extractall(testDir)
    zip_ref.close()

SAMPLE_DIR = '/data/zipped/tests_tmp'
SAMPLE_JSON = os.path.join(SAMPLE_DIR, 'test.json')
def unzipDirectory(dataDir):
    # uncompress the .zip files in the specified directory under '/data/tests_tmp'
    poolWorkerCount = 8

    if os.path.exists(SAMPLE_DIR):
        shutil.rmtree(SAMPLE_DIR)
    os.mkdir(SAMPLE_DIR)

    dataPacks = []
    for filename in os.listdir(dataDir):
        fullname = os.path.join(dataDir, filename)
        dataPacks.append((fullname, SAMPLE_DIR))

    p = Pool(poolWorkerCount)
    p.map(unzipper, dataPacks)

def genDirectoryJSON(dataDir=SAMPLE_DIR):
    import json
    filenames = [os.path.join(dataDir, fName) for fName in os.listdir(dataDir)]
    json.dump(filenames, open(SAMPLE_JSON,'w'))

def load_sample(dataDir):
    print 'Unzipping the directory %s' % dataDir
    unzipDirectory(dataDir)
    print 'Unzipping successful....'
    print 'Generating a correct JSON...'
    genDirectoryJSON()
    print 'JSON successfully generated.'

def create_conditions(args, FLAGS):
    binary_filelist = None
    # batch_size = 1
    batch_size = 32
    max_steps = 1071
    run_all = False
    visualization = None

    if args.blackout:
        args.dataDir = '/data/zipped/blackout'

    if args.dataDir=='':
        print "[ERROR] Please specify the data directory the network can load .bin files from..."
        exit(1)

    load_sample(args.dataDir)

    if args.train:  # We have 963 train patients
        if args.model == 'ae':
            binary_filelist = FLAGS.ae_train_binaries
        elif args.model == 'cae':
            binary_filelist = FLAGS.train_binaries
        else:
            binary_filelist = SAMPLE_JSON
        batch_size = 32
        # batch_size = 1
        max_steps = args.numIters
    elif args.test: # We have 108 train patients
        if args.model == 'ae':
            binary_filelist = FLAGS.ae_test_binaries
        elif args.model == 'cae':
            binary_filelist = FLAGS.test_binaries
        else:
            binary_filelist = SAMPLE_JSON
        # max_steps = 107
        max_steps = 4 #4#30#150
    elif args.blackout:
        binary_filelist = SAMPLE_JSON
        visualization = 'blackout'
        batch_size = 32
        max_steps = 33 #117*9/batchSz
    else:
        if args.model == 'ae':
            binary_filelist = FLAGS.ae_all_binaries
        elif args.model == 'cae':
            binary_filelist = FLAGS.all_binaries
        else:
            binary_filelist = SAMPLE_JSON
        run_all = True

    return binary_filelist, batch_size, max_steps, run_all, visualization

def setup_checkpoint(train, sess, saver, ckpt, ckpt_file, overrideChkpt):
    if train:
        # Get checkpoint at step: i_stopped
        if (not overrideChkpt) and ckpt and ckpt_file:
            saver.restore(sess, ckpt_file)
            print("Fetching checkpoint data from:")
            print(ckpt_file)
            i_stopped = int(ckpt_file.split('/')[-1].split('-')[-1])
        elif overrideChkpt:
            print('Overriding the checkpoints!')
            i_stopped = 0
        else:
            print('No checkpoint file found!')
            i_stopped = 0

    else: # testing (or running all files)
        # Get most recent checkpoint & start from beginning
        if ckpt and ckpt_file:
            saver.restore(sess, ckpt_file)
            print(ckpt_file)
        i_stopped = 0

    return i_stopped

def generate_CAE_output(train, run_all, encode, decode, brain_image, compressed_filelist, output_binary_filelist, FLAGS):
    # If running all files for CAE
    if not train and run_all:
        create_brain_binaries.save_and_split(compressed_filelist,
                                             output_binary_filelist,
                                             FLAGS.reduced_train_binaries,
                                             FLAGS.reduced_test_binaries)

    # Create visuals of last example
    encodeLayer = np.asarray(sess.run(encode))
    decodeLayer = np.asarray(sess.run(decode))
    inputImage = np.asarray(sess.run(brain_image))

    mat2visual(encodeLayer[0, 0,:,:,:, 0], [10, 15, 19], 'encodedImage.png', 'auto')
    mat2visual(decodeLayer[0, 0,:,:,:, 0], [40, 55, 60], 'decodedImage.png', 'auto')
    mat2visual(inputImage[0, :,:,:, 0], [40, 55, 60], 'inputImage.png', 'auto')


def create_CEA_reduced_binary(sess, encode, output, data, FLAGS, i):
    # Saving output of CAE to binary files
    encoded_image = np.asarray(sess.run(encode))
    # Get label and phenotype data
    patient_label = np.asarray(sess.run(output))
    # patient_pheno = np.asarray(sess.run(pheno_data))
    patient_pheno = np.asarray(sess.run(data))
    # ourput image currently: 31x37x31

    bin_path = create_brain_binaries.create_compressed_binary(
                            patient_pheno, encoded_image,
                            patient_label, FLAGS.reduced_dir, str(i+1))

    return bin_path


def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(df_confusion), cmap=plt.cm.jet,
                    interpolation='nearest')

    width, height = df_confusion.shape

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(df_confusion[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    labels = '01'
    plt.xticks(range(width), labels[:width])
    plt.yticks(range(height), labels[:height])
    plt.xlabel('Prediction')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png', format='png')

def compute_statistics(targets, predictions):
    conf_matrix = confusion_matrix(targets, predictions)
    accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / float(np.sum(conf_matrix))
    recall = conf_matrix[1, 1] / float(np.sum(conf_matrix[1, :]))
    precision = conf_matrix[1, 1] / float(np.sum(conf_matrix[:, 1]))
    f_score = 2 * recall * precision / (precision + recall)
    print "Accuracy of the model is: " + str(accuracy)
    print "Recall of the model is: " + str(recall)
    print "Precision of the model is: " + str(precision)
    print "F-score of the model is: " + str(f_score)

    return conf_matrix, accuracy, recall, precision, f_score

if __name__ == '__main__':
    pass