import numpy as np
import os
import json
import csv


#BRAIN_DIR = os.path.abspath('/data/pooledData')
BRAIN_DIR = os.path.abspath('/data/originalfALFFData')
PHENOTYPE_FILE = os.path.abspath('/data/processed_phenotype_data.csv')
OUTPUT_DIR = os.path.abspath('/data/binaries')
TRAIN_FILE = os.path.abspath('/data/train_brain_binary_list.json')
TEST_FILE = os.path.abspath('/data/test_brain_binary_list.json')


def create_feature_binary(X_feature, y_feature, filename):
    	# Create binary format
    	X = X_feature.flatten().tolist()
    	y = [y_feature]

    	# Label first, then 'image'
    	out = np.array(y + X, np.float32)

    	# Save
    	out.tofile(filename)


#def convert_brain_npy(brain_dir=BRAIN_DIR, phenotype_file=PHENOTYPE_FILE, output_dir=OUTPUT_DIR, pool_type='avgPool_', pool_size='_reduce2'):
def convert_brain_npy(brain_dir=BRAIN_DIR, phenotype_file=PHENOTYPE_FILE, output_dir=OUTPUT_DIR, prefix='original_'):
	path_list = []	
	with open(phenotype_file, 'r') as csvfile:
		patient_reader = csv.reader(csvfile)
		for patient in patient_reader:
			patient_id = patient[0]
			if patient_id == "":
				continue
			
			patient_label = np.float32(patient[3])
			#brain_filename = pool_type + patient_id + pool_size
			brain_filename = prefix + patient_id
			npy_filename = brain_filename + '.npy'
			bin_filename = brain_filename + '.bin' 

			npy_path = os.path.join(brain_dir, npy_filename)
			bin_path = os.path.join(output_dir, bin_filename)
			
			pooled_brain = np.load(npy_path)
			pooled_brain = pooled_brain.astype(np.float32)
			
			# create_feature_binary(pooled_brain, patient_label, bin_path)
			path_list.append(bin_path)	
	
	return path_list
	

def split_brain_binaries(file_list, split_fraction=0.9, train_file=TRAIN_FILE, test_file=TEST_FILE):
	# Error handling
	if split_fraction >= 1.0 or split_fraction <= 0.0:
		split_fration = 0.9
	
	# Calculate number of files to split
	num_files = len(file_list)
	num_train = int(num_files * split_fraction)
	
	# Permute files
    	perm = np.arange(num_files)
    	np.random.shuffle(perm)
    	file_list = [file_list[i] for i in perm]

    	# Split file list
    	train_files = file_list[:num_train]
    	test_files = file_list[num_train:]	

	#with open(TRAIN_FILE, 'w') as outfile1:
        #	json.dump(train_files, outfile1)
    	#with open(TEST_FILE, 'w') as outfile2:
        #	json.dump(test_files, outfile2)
	print train_files
	print test_files

def main():
	file_list = convert_brain_npy()
	split_brain_binaries(file_list)

if __name__ == '__main__':
	main()




