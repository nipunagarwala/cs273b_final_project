import numpy as np
import os
import json
import csv


BRAIN_DIR = os.path.abspath('/home/yinoue/pooledData')
PHENOTYPE_FILE = os.path.abspath('/data/processed_phenotype_data.csv')
OUTPUT_DIR = os.path.abspath('/data/binaries')
SAVE_FILE = os.path.abspath('/data/brain_binary_list.json')


def create_feature_binary(X_feature, y_feature, filename):
    	# Create binary format
    	X = X_feature.flatten().tolist()
    	y = [y_feature]

    	# Label first, then 'image'
    	out = np.array(y + X, np.float32)

    	# Save
    	out.tofile(filename)


def convert_brain_npy(brain_dir=BRAIN_DIR, phenotype_file=PHENOTYPE_FILE, output_dir=OUTPUT_DIR, save_file=SAVE_FILE, pool_type='avgPool_', pool_size='_reduce2'):
	path_list = []	
	with open(phenotype_file, 'r') as csvfile:
		patient_reader = csv.reader(csvfile)
		for patient in patient_reader:
			patient_id = patient[0]
			if patient_id == "":
				continue
			
			patient_label = np.float32(patient[3])
			brain_filename = pool_type + patient_id + pool_size
			npy_filename = brain_filename + '.npy'
			bin_filename = brain_filename + '.bin' 

			npy_path = os.path.join(brain_dir, npy_filename)
			bin_path = os.path.join(output_dir, bin_filename)
			
			pooled_brain = np.load(npy_path)
			pooled_brain = pooled_brain.astype(np.float32)
			
			create_feature_binary(pooled_brain, patient_label, bin_path)
			path_list.append(bin_path)	

	with open(save_file, 'w') as outfile:
		json.dump(path_list, outfile)

	# record json file containing all the binary file created
	# This will be interpreted as the filename queue later

if __name__ == '__main__':
	convert_brain_npy()
