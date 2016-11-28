  # multiple_imputation.R
 
  # This script performs random forest-based multiple imputation. This script first removes
  # features that are completely correlated with previously included features, features that
  # only serve as indicators for the data collection process (i.e. provide no clinical meaning),
  # and features that are not missing at random, for which it is not appropriate to perform
  # imputation based approaches. For the remaining features, the script then converts the null flag
  # (-9999) back into NA's. After converting some of the features to their appropriate data type,
  # the script then uses the missForest (random forest) package to impute the missing values.
  # From there, the script makes final data type conversions. 

  # By Nipun Agarwala, Yuki Inoue, Axel Sly, Olver Bear Don't Walk and David Cohn

  source(paste(code.directory, 'remove_features.R', sep = "/"))
  
  # Using Handedness scores to eliminate some of the missing handedness category
  
  if("HANDEDNESS_CATEGORY" %in% phenotype.filtered.data & 
     "HANDEDNESS_SCORES" %in% phenotype.filtered.data){
    # Vector containing row indices where handedness category data is missing
    missing.handedness.indices = which(phenotype.filtered.data$HANDEDNESS_CATEGORY == "-9999")
    
    # Dataframe containing rows of phenotype data where handedness category data is missing
    missing.handedness.phenotype.data = phenotype.filtered.data[missing.handedness.indices,]
    
    # Calculation of Handedness category based on the provided handedness score 
    # (Positive Number equals right-handed, Negative Number equals left-handed, 
    # 0 equals ambidexterous)
    missing.handedness.phenotype.data$HANDEDNESS_CATEGORY[missing.handedness.phenotype.data$HANDEDNESS_SCORES > 0] = "R"
    
    missing.handedness.phenotype.data$HANDEDNESS_CATEGORY[missing.handedness.phenotype.data$HANDEDNESS_SCORES == 0] = "Ambi"
    
    missing.handedness.phenotype.data$HANDEDNESS_CATEGORY[missing.handedness.phenotype.data$HANDEDNESS_SCORES < 0 & 
                                                            missing.handedness.phenotype.data$HANDEDNESS_SCORES != -9999] = "L"
    
    # Filling in some of missing handedness category data based on Handedness score
    phenotype.filtered.data[missing.handedness.indices,] = missing.handedness.phenotype.data
    
    phenotype.filtered.data$HANDEDNESS_CATEGORY[phenotype.filtered.data$HANDEDNESS_CATEGORY == 
                                                  "L->R"] = "-9999"
  }
 
  # Convert missing features values (as -9999) to NA's
  for (a in 1:ncol(phenotype.filtered.data)){
    phenotype.column = phenotype.filtered.data[,a]
    if(is.numeric(phenotype.column)){
      phenotype.column[phenotype.column == -9999.0] = NA
    }else if(is.integer(phenotype.column)){
      phenotype.column[phenotype.column == -9999] = NA
    }else if(is.character(phenotype.column)){
      phenotype.column[phenotype.column == "-9999"] = NA
      phenotype.column = as.factor(phenotype.column)
    }
    phenotype.filtered.data[,a] = phenotype.column
  }

  # Convert DX_GROUP, Sex, EYE_STATUS_AT_SCAN and DSM_IV_TR to factors

  if("DX_GROUP" %in% phenotype.filtered.data.column.names) {
    phenotype.filtered.data$DX_GROUP = as.factor(phenotype.filtered.data$DX_GROUP)
  }


  if("DSM_IV_TR" %in% phenotype.filtered.data.column.names) {
    phenotype.filtered.data$DSM_IV_TR = as.factor(phenotype.filtered.data$DSM_IV_TR)
  }

  if("SEX" %in% phenotype.filtered.data.column.names) {
    phenotype.filtered.data$SEX = as.factor(phenotype.filtered.data$SEX)
  }

  if("EYE_STATUS_AT_SCAN" %in% phenotype.filtered.data.column.names){
    phenotype.filtered.data$EYE_STATUS_AT_SCAN = as.factor(phenotype.filtered.data$EYE_STATUS_AT_SCAN)
  }

  # Random Forest based Multiple Imputation

  phenotype.imputed.data = missForest(phenotype.filtered.data[,-c(2,3)], ntree = 100, 
                                    verbose = TRUE, variablewise = TRUE)
  
  phenotype.imputed.names = (names(phenotype.filtered.data))[-c(2,3)]
  
  phenotype.imputed.OOB.errors = cbind.data.frame(phenotype.imputed.data$OOBerror, 
                                                  phenotype.imputed.names, 
                                                  names(phenotype.imputed.data$OOBerror))
  
  names(phenotype.imputed.OOB.errors) = c("OOB.Error", "Imputed.Feature", "OOB.Error.Type")
  
  # Separate numerical feature OOB errors from categorical feature OOB errors
  phenotype.imputed.OOB.numerical = phenotype.imputed.OOB.errors %>%
    dplyr::filter(OOB.Error.Type == "MSE")
  
  phenotype.imputed.OOB.categorical = phenotype.imputed.OOB.errors %>%
    dplyr::filter(OOB.Error.Type == "PFC")
  
  # Plot MSE for Continuous phenotype features
  
  phenotype.imputed.OOB.numerical$Imputed.Feature = c("Age", "HANDED_Score", "FIQ",
                                                      "VIQ", "PIQ", "SRS_Aware",
                                                      "SRS_Cog", "SRS_Com", "SRS_Mot",
                                                      "SRS_Man", "VINE_Com", "VINE_Dail",
                                                      "VINE_Soc", "VINE_ABC", "WISC_VCI",
                                                      "WISC_PRI", "WISC_WMI", "WISC_PSI", "BMI")
  
  names(phenotype.imputed.OOB.numerical) = c("MSE.Error", "Imputed.Feature", "OOB.Error.Type")
  
  phenotype.imputed.OOB.numerical.plot = ggplot(phenotype.imputed.OOB.numerical,
                                                aes(Imputed.Feature, MSE.Error)) + 
    geom_bar(stat = "identity") + ylim(0, 100) + 
    theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
    ggtitle("Imputation OOB MSE Error for Continuous Variables") + ylab("Mean Squared Error")
  
  print(phenotype.imputed.OOB.numerical.plot)
  
  # Plot Proportion of Falsely Classified Values (PFC) for Categorical Variables
  
  phenotype.imputed.OOB.categorical$Imputed.Feature = c("SITE_ID", "Sex", "HAND_Cat", "FIQ_Type",
                                                        "VIQ_Type", "PIQ_Type", "MED_STATUS", 
                                                        "EYE_STATUS")
  
  names(phenotype.imputed.OOB.categorical) = c("PFC.Error", "Imputed.Feature", "OOB.Error.Type")
  
  phenotype.imputed.OOB.categorical.plot = ggplot(phenotype.imputed.OOB.categorical,
                                                  aes(Imputed.Feature, PFC.Error)) + 
    geom_bar(stat = "identity") + theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
    ggtitle("Imputation OOB PFC Error for Categorical Variables") + 
    ylab("Proportion of Falsely Classified")
  
  print(phenotype.imputed.OOB.categorical.plot)
  
  # Combining SITE_ID, SUB_ID and disease group, and imputed data in a single data frame
  phenotype.filtered.data = cbind((phenotype.imputed.data$ximp)[,1], # SITE_ID,
                                phenotype.filtered.data[,2:3], # SUB_ID and disease status,
                                phenotype.imputed.data$ximp[,-1]) # imputed data

  phenotype.filtered.data.column.names = names(phenotype.filtered.data)

  phenotype.filtered.data.column.names[1] = "SITE_ID"

  names(phenotype.filtered.data) = phenotype.filtered.data.column.names

  if("CURRENT_MED_STATUS" %in% phenotype.filtered.data.column.names){
    missing.med.status = which(phenotype.filtered.data$CURRENT_MED_STATUS == '`')
    if(length(missing.med.status) > 0){
      phenotype.filtered.data$CURRENT_MED_STATUS[missing.med.status] = 0
    }
  }
 
  source(paste(code.directory, 'convert_test_scores.R', sep = "/"))