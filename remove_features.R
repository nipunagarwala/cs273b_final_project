  # remove_features.R

  # This method removes that are completely correlated with previously included features, 
  # features that only serve as indicators for the data collection process (i.e. provide no 
  # clinical meaning), features that are too sparse, and features that are not missing at random,
  # for which it is not appropriate to perform imputation based approaches.

  # By Nipun Agarwala, Yuki Inoue, Axel Sly, Olver Bear Don't Walk and David Cohn

  phenotype.filtered.data.column.names = names(phenotype.filtered.data)

  # Remove Autism Diagnostic Interview (ADI features, autism-specific) 
  ADI.indices = grep("ADI", phenotype.filtered.data.column.names)

  if(length(ADI.indices) > 0){
    phenotype.filtered.data = phenotype.filtered.data[,-ADI.indices]
    phenotype.filtered.data.column.names = names(phenotype.filtered.data)
  }
  
  # Remove Autism Diagnostic Observation Schedule Features (ADOS features, autism-specific)
  ADOS.indices = grep("ADOS", phenotype.filtered.data.column.names)

  if(length(ADOS.indices) > 0){
    phenotype.filtered.data = phenotype.filtered.data[, -ADOS.indices]
    phenotype.filtered.data.column.names = names(phenotype.filtered.data)
  }

  # Remove string-based features (Comorbidity and Medication Name) from feature matrix 
  # (too sparse)
  comorbidity.index = grep("COMORBIDITY", phenotype.filtered.data.column.names)

  if(comorbidity.index != 0){
    phenotype.filtered.data = phenotype.filtered.data[, -comorbidity.index]
    phenotype.filtered.data.column.names = names(phenotype.filtered.data)
  }

  medication.name.index = grep("MEDICATION_NAME", phenotype.filtered.data.column.names)

  if(medication.name.index != 0){
    phenotype.filtered.data = phenotype.filtered.data[, -medication.name.index]
    phenotype.filtered.data.column.names = names(phenotype.filtered.data)
  }

  # Remove Off Stimulants at scan feature (no clinical relevance)
  off.stimulants.index = grep("OFF_STIMULANTS_AT_SCAN", phenotype.filtered.data.column.names)

  if(off.stimulants.index != 0){
    phenotype.filtered.data = phenotype.filtered.data[, -off.stimulants.index]
    phenotype.filtered.data.column.names = names(phenotype.filtered.data)
  }

  # Remove VINELAND and WISC-IV scaled features 
  # (VINELAND and WISC-IV standard features already included in filtered data set)
  scaled.features.indices = grep("SCALED", phenotype.filtered.data.column.names)

  if(length(scaled.features.indices) > 0) {
    phenotype.filtered.data = phenotype.filtered.data[, -scaled.features.indices]
    phenotype.filtered.data.column.names = names(phenotype.filtered.data)
  }

  # Remove SRS_TOTAL, SCQ_TOTAL, and AQ_TOTAL (correlated with previously included features)
  total.indices = grep("TOTAL", phenotype.filtered.data.column.names)

  if(length(total.indices) > 0){
    phenotype.filtered.data = phenotype.filtered.data[, -total.indices]
    phenotype.filtered.data.column.names = names(phenotype.filtered.data)
  }
  
  # Remove Age at Anatomical Scan in Years feature (correlated with previously included feature)
  AGE.AT.MPRAGE.index = grep("AGE_AT_MPRAGE", phenotype.filtered.data.column.names)

  if(AGE.AT.MPRAGE.index != 0){
    phenotype.filtered.data = phenotype.filtered.data[, -AGE.AT.MPRAGE.index]
    phenotype.filtered.data.column.names = names(phenotype.filtered.data)
  }
  
  # Remove SRS_Version Feature (not clinically relevant)
  SRS.VERSION.index = grep("SRS_VERSION", phenotype.filtered.data.column.names)

  if(SRS.VERSION.index != 0){
    phenotype.filtered.data = phenotype.filtered.data[, -SRS.VERSION.index]
    phenotype.filtered.data.column.names = names(phenotype.filtered.data)
  }
  
  # Remove VINELAND_INFORMANT Feature (not clinically relevant)
  VINELAND.INFORMANT.index = grep("INFORMANT", phenotype.filtered.data.column.names)

  if(VINELAND.INFORMANT.index != 0){
    phenotype.filtered.data = phenotype.filtered.data[, -VINELAND.INFORMANT.index]
    phenotype.filtered.data.column.names = names(phenotype.filtered.data)
  }
  
  # Remove VINELAND_SUM Feature (Correlated with previously included features) 
  VINELAND.SUM.index = grep("VINELAND_SUM_SCORES", phenotype.filtered.data.column.names)

  if(VINELAND.SUM.index != 0){
    phenotype.filtered.data = phenotype.filtered.data[, -VINELAND.SUM.index]
    phenotype.filtered.data.column.names = names(phenotype.filtered.data)
  }
  
  # Remove DSM_IV_index (Correlated with previously included features)
  DSM.IV.index = grep("DSM_IV", phenotype.filtered.data.column.names)

  if(DSM.IV.index != 0){
    phenotype.filtered.data = phenotype.filtered.data[, -DSM.IV.index]
    phenotype.filtered.data.column.names = names(phenotype.filtered.data)
  }