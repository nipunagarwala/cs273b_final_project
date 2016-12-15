# import create_brain_binaries as c
#
# c.convert_random_brain_npy('/data/swap_partial_binaries', 'adgsgag', True)
#
# import os
# import numpy as np
#
# LABEL_SZ = 1
# PHENO_SZ = 29
# X_SZ = 31
# Y_SZ = 37
# Z_SZ = 31
# dirName = '/data/swap_partial_binaries_reduced' #_reduced
# count = 0
# for filename in os.listdir(dirName):
#     string_label = filename.split("_")[2] #[1]
#     actual_label = 1 if string_label == "autism" else 0
#
#     # print os.path.join(dirName, filename)
#     file_label = int(np.memmap(filename=os.path.join(dirName,filename), dtype='float32',
#                       mode='r', offset=0, shape=1))
#
#     print (actual_label == file_label)
#     # print (actual_label, file_label)
#     # print actual_label == file_label
#
#     if actual_label != file_label:
#         count += 1
#     count +=1
#
# print count #/ float(len(os.listdir(dirName)))

# import json
#
# with open('/data/swap_partial_train_reduced.json') as f:
#     filelist = json.load(f)
#     f = set(filelist)
#     print len(f)
#     print len(filelist)
#     print [c for c in filelist if "555" in c]
#     # for path in filelist:
#     #     print path


import numpy as np

path = '/data/augmented_swap_partial/489_autism_partially_patched_9.npy'
a = np.load(path)
print a
