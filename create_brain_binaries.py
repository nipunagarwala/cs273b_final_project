import numpy as np
import os
import json
import csv


BRAIN_DIR = os.path.abspath('/data/originalfALFFData')
# PHENOTYPE_FILE = os.path.abspath('/data/processed_imputed_phenotype_data.csv')
PHENOTYPE_FILE = os.path.abspath('/data/normalized_processed_imputed_phenotype_data.csv')
# OUTPUT_DIR = os.path.abspath('/data/binaries')
OUTPUT_DIR = os.path.abspath('/data/binaries_new')
# ALL_FILES = os.path.abspath('/data/brain_binary_list.json')
# TRAIN_FILE = os.path.abspath('/data/train_brain_binary_list.json')
# TEST_FILE = os.path.abspath('/data/test_brain_binary_list.json')
ALL_FILES = os.path.abspath('/data/all.json')
TRAIN_FILE = os.path.abspath('/data/train.json')
TEST_FILE = os.path.abspath('/data/test.json')


def _create_feature_binary(X_data, X_image, y_feature, filename):
    # Create binary format
    X_1 = X_data.flatten().tolist()
    X_2 = X_image.flatten().tolist()
    y = [y_feature]

    # Label first, then 'image'
    out = np.array(y + X_1 + X_2, np.float32)

    # Save
    out.tofile(filename)


def _normalize_brain(brain_data):
    # Get mean and variance
    mean = brain_data.mean()
    std_dev = brain_data.std()

    # Normalize
    return (brain_data - mean)/brain_data.std()


def create_compressed_binary(phenotype, image, label, output_dir, id, prefix='compressed_'):
    bin_filename = prefix + id + '.bin'
    bin_path = os.path.join(output_dir, bin_filename)
    _create_feature_binary(phenotype, image, label, bin_path)
    return bin_path


def convert_brain_npy(brain_dir=BRAIN_DIR, phenotype_file=PHENOTYPE_FILE, output_dir=OUTPUT_DIR, prefix='original_'):
    path_list = []
    with open(phenotype_file, 'r') as csvfile:
        patient_reader = csv.reader(csvfile)
        for patient in patient_reader:
            patient_id = patient[0]
            if patient_id == "":
                continue

            # Retrieve data from phenotype CSV
            patient_label = np.float32(patient[3])
            phenotype_data = patient[5:16] + patient[19:]
            phenotype_data = np.asarray(phenotype_data, dtype=np.float32)
            # Normalize phenotype data?

            # Create necessary file names
            # brain_filename = pool_type + patient_id + pool_size
            brain_filename = prefix + patient_id
            npy_filename = brain_filename + '.npy'
            bin_filename = patient_id + '.bin'

            # Create necessary paths
            npy_path = os.path.join(brain_dir, npy_filename)
            bin_path = os.path.join(output_dir, bin_filename)

            # Load brain images from .npy files
            brain_data = np.load(npy_path)
            brain_data = brain_data.astype(np.float32)
            normailized_brain = _normalize_brain(brain_data)

            # Create binaries from all data
            _create_feature_binary(phenotype_data, normailized_brain, patient_label, bin_path)
            path_list.append(bin_path)

    return path_list


def split_brain_binaries(file_list, train_file, test_file, split_fraction=0.9):
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

	with open(train_file, 'w') as outfile1:
		json.dump(train_files, outfile1)

	with open(test_file, 'w') as outfile2:
		json.dump(test_files, outfile2)


def save_and_split(file_list, all_files=ALL_FILES, train_file=TRAIN_FILE, test_file=TEST_FILE):
    with open(all_files, 'w') as outfile:
        json.dump(file_list, outfile)
	split_brain_binaries(file_list, train_file, test_file)


def main():
    file_list = convert_brain_npy()
    save_and_split(file_list)

if __name__ == '__main__':
	main()
