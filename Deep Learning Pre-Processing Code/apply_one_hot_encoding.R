  # apply_one_hot_encoding.R

  # This script applies one-hot encoding to categorical variables. 

  # By Nipun Agarwala, Yuki Inoue, Axel Sly, Olver Bear Don't Walk and David Cohn

  # Adding SITE_ID and patient SUB_ID to dataframe with one-hot encoded categorical variables
  phenotype.encoded.data=phenotype.filtered.data[,1:2]

  for (i in 3:ncol(phenotype.filtered.data)){
  
    if(names(phenotype.filtered.data)[i] == "DX_GROUP"){
    
      dummy.variable.data=phenotype.filtered.data %>%
        dplyr::select(SUB_ID, DX_GROUP) %>%
        mutate(DX_GROUP_Autism=ifelse(DX_GROUP == 1, 1, 0)) %>%
        mutate(DX_GROUP_Control=ifelse(DX_GROUP == 2, 1, 0)) %>%
        # Verify if values are missing for both categories
        mutate(DX_GROUP_missing_boolean = DX_GROUP_Autism + DX_GROUP_Control) %>%
        dplyr::select(SUB_ID, DX_GROUP_Autism, DX_GROUP_Control, DX_GROUP_missing_boolean)
    
      dummy.variable.missing.indices = which(dummy.variable.data$DX_GROUP_missing_boolean == 0)
      # Add -9999 (indicating missing values with no one-hot encoding)
      if(length(dummy.variable.missing.indices) > 0) {
        for (z in 1:length(dummy.variable.missing.indices)){
          row.index = dummy.variable.missing.indices[z]
          dummy.variable.data[row.index, 2] = -9999
          dummy.variable.data[row.index, 3] = -9999
        }
      }
      dummy.variable.data = dplyr::select(dummy.variable.data, 
                                        SUB_ID, DX_GROUP_Autism, DX_GROUP_Control)
      phenotype.encoded.data=inner_join(phenotype.encoded.data, dummy.variable.data, 
                                        by="SUB_ID")
    
    }else if(names(phenotype.filtered.data)[i] == "SEX"){
    
      dummy.variable.data=phenotype.filtered.data %>%
        dplyr::select(SUB_ID, SEX) %>%
        mutate(SEX_Male=ifelse(SEX == 1, 1, 0)) %>%
        mutate(SEX_Female=ifelse(SEX == 2, 1, 0)) %>%
        # Verify if values are missing (0) for both categories
        mutate(SEX_missing_boolean = SEX_Female + SEX_Male) %>%
        dplyr::select(SUB_ID, SEX_Male, SEX_Female, SEX_missing_boolean)
    
      dummy.variable.missing.indices = which(dummy.variable.data$SEX_missing_boolean == 0)
      
      # Add -9999 (indicating missing values with no one-hot encoding)
      if(length(dummy.variable.missing.indices) > 0) {
        for (z in 1:length(dummy.variable.missing.indices)){
          row.index = dummy.variable.missing.indices[z]
          dummy.variable.data[row.index, 2] = -9999
          dummy.variable.data[row.index, 3] = -9999
        }
      }
      dummy.variable.data = dplyr::select(dummy.variable.data, 
                                        SUB_ID, SEX_Male, SEX_Female)
    
      phenotype.encoded.data=inner_join(phenotype.encoded.data, dummy.variable.data, 
                                        by="SUB_ID")
    
    }else if(names(phenotype.filtered.data)[i] == "SRS_VERSION") {
    
      dummy.variable.data=phenotype.filtered.data %>%
        dplyr::select(SUB_ID, SRS_VERSION) %>%
        mutate(SRS_child=ifelse(SRS_VERSION == 1, 1, 0)) %>%
        mutate(SRS_adult=ifelse(SRS_VERSION == 2, 1, 0)) %>%
        # Verify if values are missing (0) for both categories
        mutate(SRS_missing_boolean = SRS_child + SRS_adult) %>%
        dplyr::select(SUB_ID, SRS_child, SRS_adult, SRS_missing_boolean)
    
      dummy.variable.missing.indices = which(dummy.variable.data$SRS_missing_boolean == 0)
      
      # Add -9999 (indicating missing values with no one-hot encoding)
      if(length(dummy.variable.missing.indices) > 0) {
        for (z in 1:length(dummy.variable.missing.indices)){
          row.index = dummy.variable.missing.indices[z]
          dummy.variable.data[row.index, 2] = -9999
          dummy.variable.data[row.index, 3] = -9999
        }
      }
      dummy.variable.data = dplyr::select(dummy.variable.data, 
                                        SUB_ID, SRS_child, SRS_adult)
    
      phenotype.encoded.data=inner_join(phenotype.encoded.data, dummy.variable.data, 
                                        by="SUB_ID")
    
    }else if(names(phenotype.filtered.data)[i] == "VINELAND_INFORMANT"){
    
      dummy.variable.data=phenotype.filtered.data %>%
        dplyr::select(SUB_ID, VINELAND_INFORMANT) %>%
        mutate(VINELAND_parent=ifelse(VINELAND_INFORMANT == 1, 1, 0)) %>%
        mutate(VINELAND_self=ifelse(VINELAND_INFORMANT == 2, 1, 0)) %>%
        # Verify if values are missing (0) for both categories
        mutate(VINELAND_missing_boolean = VINELAND_parent + VINELAND_self) %>%
        dplyr::select(SUB_ID, VINELAND_parent, VINELAND_self, VINELAND_missing_boolean)
    
      dummy.variable.missing.indices = which(dummy.variable.data$VINELAND_missing_boolean == 0)
      # Add -9999 (indicating missing values with no one-hot encoding)
      if(length(dummy.variable.missing.indices) > 0) {
        for (z in 1:length(dummy.variable.missing.indices)){
          row.index = dummy.variable.missing.indices[z]
          dummy.variable.data[row.index, 2] = -9999
          dummy.variable.data[row.index, 3] = -9999
        }
      }
    dummy.variable.data = dplyr::select(dummy.variable.data, 
                                        SUB_ID, VINELAND_parent, VINELAND_self)
    
    phenotype.encoded.data=inner_join(phenotype.encoded.data, dummy.variable.data, 
                                      by="SUB_ID")
    
    }else if(names(phenotype.filtered.data)[i] == "EYE_STATUS_AT_SCAN") {
    
      dummy.variable.data=phenotype.filtered.data %>%
        dplyr::select(SUB_ID, EYE_STATUS_AT_SCAN) %>%
        mutate(EYE_STATUS_AT_SCAN_open=ifelse(EYE_STATUS_AT_SCAN == 1, 1, 0)) %>%
        mutate(EYE_STATUS_AT_SCAN_closed=ifelse(EYE_STATUS_AT_SCAN == 2, 1, 0)) %>%
        # Verify if values are missing (0) for both categories
        mutate(EYE_status_missing_boolean = EYE_STATUS_AT_SCAN_open + EYE_STATUS_AT_SCAN_closed) %>%
        dplyr::select(SUB_ID, EYE_STATUS_AT_SCAN_open, EYE_STATUS_AT_SCAN_closed, 
                      EYE_status_missing_boolean)
    
      dummy.variable.missing.indices = which(dummy.variable.data$EYE_status_missing_boolean == 0)
      
      # Add -9999 (indicating missing values with no one-hot encoding)
      if(length(dummy.variable.missing.indices) > 0) {
        for (z in 1:length(dummy.variable.missing.indices)){
          row.index = dummy.variable.missing.indices[z]
          dummy.variable.data[row.index, 2] = -9999
          dummy.variable.data[row.index, 3] = -9999
        }
      }
    
      dummy.variable.data = dplyr::select(dummy.variable.data, 
                                        SUB_ID, EYE_STATUS_AT_SCAN_open, 
                                        EYE_STATUS_AT_SCAN_closed)
    
      phenotype.encoded.data=inner_join(phenotype.encoded.data, dummy.variable.data, 
                                        by="SUB_ID")
    
    }else if(names(phenotype.filtered.data)[i]=="DSM_IV_TR"){
    
      dummy.variable.data=phenotype.filtered.data %>%
        dplyr::select(SUB_ID, DSM_IV_TR) %>%
        mutate(DSM_IV_TR_control=ifelse(DSM_IV_TR == 0, 1, 0)) %>%
        mutate(DSM_IV_TR_autism=ifelse(DSM_IV_TR == 1, 1, 0)) %>%
        mutate(DSM_IV_TR_aspergers=ifelse(DSM_IV_TR == 2, 1, 0)) %>%
        mutate(DSM_IV_TR_PDD_NOS=ifelse(DSM_IV_TR == 3, 1, 0)) %>%
        mutate(DSM_IV_TR_aspergers_PDD_NOS=ifelse(DSM_IV_TR == 4, 1, 0)) %>%
        # Verify if values are missing (0) for all categories
        mutate(DSM_IV_missing_boolean = DSM_IV_TR_control + DSM_IV_TR_autism + DSM_IV_TR_aspergers + 
               DSM_IV_TR_PDD_NOS + DSM_IV_TR_aspergers_PDD_NOS) %>%
        dplyr::select(SUB_ID, DSM_IV_TR_control, DSM_IV_TR_autism, DSM_IV_TR_aspergers, 
                    DSM_IV_TR_PDD_NOS, DSM_IV_TR_aspergers_PDD_NOS, DSM_IV_missing_boolean)
    
      dummy.variable.missing.indices = which(dummy.variable.data$DSM_IV_missing_boolean == 0)
      
      # Add -9999 (indicating missing values with no one-hot encoding)
      if(length(dummy.variable.missing.indices) > 0) {
        for (z in 1:length(dummy.variable.missing.indices)){
          row.index = dummy.variable.missing.indices[z]
          dummy.variable.data[row.index, 2] = -9999
          dummy.variable.data[row.index, 3] = -9999
          dummy.variable.data[row.index, 4] = -9999
          dummy.variable.data[row.index, 5] = -9999
          dummy.variable.data[row.index, 6] = -9999
        }
      }
      dummy.variable.data = dplyr::select(dummy.variable.data, 
                                        SUB_ID, DSM_IV_TR_control, DSM_IV_TR_autism, 
                                        DSM_IV_TR_aspergers, DSM_IV_TR_PDD_NOS, 
                                        DSM_IV_TR_aspergers_PDD_NOS)
    
      phenotype.encoded.data=inner_join(phenotype.encoded.data, dummy.variable.data, 
                                      by="SUB_ID")
    
    }else if(names(phenotype.filtered.data)[i]=="HANDEDNESS_CATEGORY"){
    
      dummy.variable.data=phenotype.filtered.data %>%
        dplyr::select(SUB_ID, HANDEDNESS_CATEGORY) %>%
        mutate(HANDEDNESS_CATEGORY_right=ifelse(HANDEDNESS_CATEGORY=="R", 1, 0)) %>%
        mutate(HANDEDNESS_CATEGORY_left=ifelse(HANDEDNESS_CATEGORY=="L", 1, 0)) %>%
        mutate(HANDEDNESS_CATEGORY_ambi=ifelse(HANDEDNESS_CATEGORY=="Ambi", 1, 0)) %>%
        mutate(HANDEDNESS_CATEGORY_mixed=ifelse(HANDEDNESS_CATEGORY == "Mixed", 1, 0)) %>%
        # Verify if values are missing (0) for all categories
        mutate(Handedness_missing_boolean = HANDEDNESS_CATEGORY_right + HANDEDNESS_CATEGORY_left + 
               HANDEDNESS_CATEGORY_ambi + HANDEDNESS_CATEGORY_mixed) %>%
        dplyr::select(SUB_ID, HANDEDNESS_CATEGORY_right, HANDEDNESS_CATEGORY_left, 
                    HANDEDNESS_CATEGORY_ambi, Handedness_missing_boolean, HANDEDNESS_CATEGORY_mixed)
    
      dummy.variable.missing.indices = which(dummy.variable.data$Handedness_missing_boolean == 0)
    
      # Add -9999 (indicating missing values with no one-hot encoding)
      if(length(dummy.variable.missing.indices) > 0) {
        for (z in 1:length(dummy.variable.missing.indices)){
          row.index = dummy.variable.missing.indices[z]
          dummy.variable.data[row.index, 2] = -9999
          dummy.variable.data[row.index, 3] = -9999
          dummy.variable.data[row.index, 4] = -9999
        }
      }
      dummy.variable.data = dplyr::select(dummy.variable.data, SUB_ID, HANDEDNESS_CATEGORY_right, 
                                        HANDEDNESS_CATEGORY_left, HANDEDNESS_CATEGORY_ambi, 
                                        HANDEDNESS_CATEGORY_mixed)
      phenotype.encoded.data=inner_join(phenotype.encoded.data, dummy.variable.data, by="SUB_ID")
    
    }else if(names(phenotype.filtered.data)[i]=="ADI_R_RSRCH_RELIABLE"){
    
      dummy.variable.data=phenotype.filtered.data %>%
        dplyr::select(SUB_ID, ADI_R_RSRCH_RELIABLE) %>%
        mutate(ADI_R_RSRCH_RELIABLE_not_reliable=ifelse(ADI_R_RSRCH_RELIABLE==0, 1, 0)) %>%
        mutate(ADI_R_RSRCH_RELIABLE_reliable=ifelse(ADI_R_RSRCH_RELIABLE==1, 1, 0)) %>%
        # Verify if values are missing (0) for all categories
        mutate(AD_R_RSRCH_missing_boolean = ADI_R_RSRCH_RELIABLE_not_reliable + 
               ADI_R_RSRCH_RELIABLE_reliable) %>%
        dplyr::select(SUB_ID, ADI_R_RSRCH_RELIABLE_not_reliable, 
                    ADI_R_RSRCH_RELIABLE_reliable, AD_R_RSRCH_missing_boolean)
    
      dummy.variable.missing.indices = which(dummy.variable.data$AD_R_RSRCH_missing_boolean == 0)
      
      # Add -9999 (indicating missing values with no one-hot encoding)
      if(length(dummy.variable.missing.indices) > 0) {
        for (z in 1:length(dummy.variable.missing.indices)){
          row.index = dummy.variable.missing.indices[z]
          dummy.variable.data[row.index, 2] = -9999
          dummy.variable.data[row.index, 3] = -9999
        }
      }
      dummy.variable.data = dplyr::select(dummy.variable.data, SUB_ID, 
                                  ADI_R_RSRCH_RELIABLE_not_reliable, ADI_R_RSRCH_RELIABLE_reliable)
      phenotype.encoded.data=inner_join(phenotype.encoded.data, dummy.variable.data, 
                                      by="SUB_ID")
    
    }else if(names(phenotype.filtered.data)[i]=="CURRENT_MED_STATUS"){
    
      dummy.variable.data=phenotype.filtered.data %>%
        dplyr::select(SUB_ID, CURRENT_MED_STATUS) %>%
        mutate(CURRENT_MED_STATUS_no_med=ifelse(CURRENT_MED_STATUS==0, 1, 0)) %>%
        mutate(CURRENT_MED_STATUS_med=ifelse(CURRENT_MED_STATUS==1, 1, 0)) %>%
        # Verify if values are missing (0) for all categories
        mutate(CURRENT_MED_missing_boolean = CURRENT_MED_STATUS_no_med + CURRENT_MED_STATUS_med) %>%
        dplyr::select(SUB_ID, CURRENT_MED_STATUS_no_med, CURRENT_MED_STATUS_med, 
                    CURRENT_MED_missing_boolean)
    
      dummy.variable.missing.indices = which(dummy.variable.data$CURRENT_MED_missing_boolean == 0)
      
      # Add -9999 (indicating missing values with no one-hot encoding)
      if(length(dummy.variable.missing.indices) > 0) {
        for (z in 1:length(dummy.variable.missing.indices)){
          row.index = dummy.variable.missing.indices[z]
          dummy.variable.data[row.index, 2] = -9999
          dummy.variable.data[row.index, 3] = -9999
        }
      }
      dummy.variable.data = dplyr::select(dummy.variable.data, SUB_ID, 
                                        CURRENT_MED_STATUS_no_med, CURRENT_MED_STATUS_med)
    
      phenotype.encoded.data=inner_join(phenotype.encoded.data, dummy.variable.data, by="SUB_ID")
    
    }else if(names(phenotype.filtered.data)[i]=="OFF_STIMULANTS_AT_SCAN"){
    
      dummy.variable.data=phenotype.filtered.data %>%
        dplyr::select(SUB_ID, OFF_STIMULANTS_AT_SCAN) %>%
        mutate(OFF_STIMULANTS_AT_SCAN_no=ifelse(OFF_STIMULANTS_AT_SCAN==0, 1, 0)) %>%
        mutate(OFF_STIMULANTS_AT_SCAN_yes=ifelse(OFF_STIMULANTS_AT_SCAN==1, 1, 0)) %>%
        # Verify if values are missing (0) for all categories
        mutate(OFF_STIMULANTS_missing_boolean = OFF_STIMULANTS_AT_SCAN_no + OFF_STIMULANTS_AT_SCAN_yes) %>%
        dplyr::select(SUB_ID, OFF_STIMULANTS_AT_SCAN_no, OFF_STIMULANTS_AT_SCAN_yes, 
                    OFF_STIMULANTS_missing_boolean)
    
      dummy.variable.missing.indices = which(dummy.variable.data$OFF_STIMULANTS_missing_boolean == 0)
      
      # Add -9999 (indicating missing values with no one-hot encoding)
      if(length(dummy.variable.missing.indices) > 0) {
        for (z in 1:length(dummy.variable.missing.indices)){
          row.index = dummy.variable.missing.indices[z]
          dummy.variable.data[row.index, 2] = -9999
          dummy.variable.data[row.index, 3] = -9999
        }
      }
      dummy.variable.data = dplyr::select(dummy.variable.data, SUB_ID, 
                                        OFF_STIMULANTS_AT_SCAN_no, OFF_STIMULANTS_AT_SCAN_yes)
    
      phenotype.encoded.data=inner_join(phenotype.encoded.data, dummy.variable.data, by="SUB_ID")
    
    }else{
      # Adding non-categorical variable to dataframe with one-hot encoded categorical values
      variable.data=phenotype.filtered.data[,c(2,i)]
      phenotype.encoded.data=inner_join(phenotype.encoded.data, variable.data, by="SUB_ID")
    }
  }
