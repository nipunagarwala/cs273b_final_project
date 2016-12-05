  # deep_learning_pre_processing.R

  # This script serves as the "main" file that contains the scripts that process the phenotype, 
  # fALFF and ROI data associated with the ABIDE Autism data set. Script files within this 
  # overarching file will describe in greater detail the methodologies used to pre-process the 
  # various data sources above.

  # By Nipun Agarwala, Yuki Inoue, Axel Sly, Olver Bear Don't Walk and David Cohn

  # Libraries used in pre-processing scripts
  library(knitr)
  library(tidyr)
  library(dplyr)
  library(missForest)
  library(ggplot2)
  library(caret)
  library(glmnet)
  library(pROC)

  # Directory that contains script files
  code.directory = '/Users/davidcohniii/Documents/CS 273 Code and Packets'

  # Directory that contains fALFF and/or ROI data sources
  data.directory = '/Users/davidcohniii/Documents/CS 273 Code and Packets'
  
  # Directory that stores processed fALFF data
  
  fALFF.directory = '/Users/davidcohniii/Documents/fALFF Data'
  
  # Directory that stores processed ROI data
  
  ROI.directory = '/Users/davidcohniii/Documents/ROI Data'

  setwd(data.directory)

  if(file.exists('ABIDE_fALFF_2.RData')){
    fALFF.data = load('ABIDE_fALFF_2.RData')
    source(paste(code.directory, 'process_phenotype_data.R', sep = "/"))
    source(paste(code.directory, 'process_fALFF_data.R', sep = "/"))
    setwd(code.directory)
    source(paste(code.directory, 'lasso_logistic_regression.R', sep = "/"))
  }
  if(file.exists('ABIDE_AAL_116_ROI.RData')) {
    ROI.data = load('ABIDE_AAL_116_ROI.RData')
    source(paste(code.directory, 'process_ROI_phenotype_data.R', sep = "/"))
    source(paste(code.directory, 'process_AAL_ROI_data.R', sep = "/"))
    setwd(code.directory)
  }