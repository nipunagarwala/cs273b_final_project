# swapout_generator.py - generates augmentated dataset

from utils import zipDirectory
from utils_fMRI_augment import prepreProcess,augmentPartialSwap
from cae_run import applyCAE

RUN_ALL = 2

if __name__ == '__main__':
	num_swapout = 58

	#prepreProcess()
	# generate the dataset in bin, with proper feature
	#augmentPartialSwap(num_swapout, groupType='autism')
	# reduce the data using the CAE
	#applyCAE(state=RUN_ALL, input_dir='/data/augmented_swap_partial_steal_%d' % num_swapout)
	# zip 'em up
	zipDirectory('/data/augmented_swap_partial_steal_%d_reduced' % num_swapout, 
	 			 outputDirName='/data/zipped/swap%d' % num_swapout)
