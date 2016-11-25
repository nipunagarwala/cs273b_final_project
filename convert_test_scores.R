# convert_test_scores.R

# This method converts imputed behavioral and IQ test scores from type numeric to 
# type integer.

# By Nipun Agarwala, Yuki Inoue, Axel Sly, Olver Bear Don't Walk and David Cohn

if("FIQ" %in% phenotype.filtered.data.column.names){
  phenotype.filtered.data$FIQ = as.integer(phenotype.filtered.data$FIQ)
}

if("VIQ" %in% phenotype.filtered.data.column.names){
  phenotype.filtered.data$VIQ = as.integer(phenotype.filtered.data$VIQ)
}

if("PIQ" %in% phenotype.filtered.data.column.names){
  phenotype.filtered.data$PIQ = as.integer(phenotype.filtered.data$PIQ)
}

if("SRS_AWARENESS" %in% phenotype.filtered.data.column.names){
  phenotype.filtered.data$SRS_AWARENESS = as.integer(phenotype.filtered.data$SRS_AWARENESS)
}

if("SRS_COGNITION" %in% phenotype.filtered.data.column.names){
  phenotype.filtered.data$SRS_COGNITION = as.integer(phenotype.filtered.data$SRS_COGNITION)
}

if("SRS_COMMUNICATION" %in% phenotype.filtered.data.column.names){
  phenotype.filtered.data$SRS_COMMUNICATION = as.integer(phenotype.filtered.data$SRS_COMMUNICATION)
}

if("SRS_MOTIVATION" %in% phenotype.filtered.data.column.names){
  phenotype.filtered.data$SRS_MOTIVATION = as.integer(phenotype.filtered.data$SRS_MOTIVATION)
}

if("SRS_MANNERISMS" %in% phenotype.filtered.data.column.names){
  phenotype.filtered.data$SRS_MANNERISMS = as.integer(phenotype.filtered.data$SRS_MANNERISMS)
}

if("VINELAND_COMMUNICATION_STANDARD" %in% phenotype.filtered.data.column.names){
  phenotype.filtered.data$VINELAND_COMMUNICATION_STANDARD = as.integer(phenotype.filtered.data$VINELAND_COMMUNICATION_STANDARD)
}

if("VINELAND_DAILYLVNG_STANDARD" %in% phenotype.filtered.data.column.names){
  phenotype.filtered.data$VINELAND_DAILYLVNG_STANDARD = as.integer(phenotype.filtered.data$VINELAND_DAILYLVNG_STANDARD)
}

if("VINELAND_SOCIAL_STANDARD" %in% phenotype.filtered.data.column.names){
  phenotype.filtered.data$VINELAND_SOCIAL_STANDARD = as.integer(phenotype.filtered.data$VINELAND_SOCIAL_STANDARD)
}

if("VINELAND_ABC_STANDARD" %in% phenotype.filtered.data.column.names){
  phenotype.filtered.data$VINELAND_ABC_STANDARD = as.integer(phenotype.filtered.data$VINELAND_ABC_STANDARD)
}

if("WISC_IV_VCI" %in% phenotype.filtered.data.column.names){
  phenotype.filtered.data$WISC_IV_VCI = as.integer(phenotype.filtered.data$WISC_IV_VCI)
}

if("WISC_IV_PRI" %in% phenotype.filtered.data.column.names){
  phenotype.filtered.data$WISC_IV_PRI = as.integer(phenotype.filtered.data$WISC_IV_PRI)
}

if("WISC_IV_WMI" %in% phenotype.filtered.data.column.names){
  phenotype.filtered.data$WISC_IV_WMI = as.integer(phenotype.filtered.data$WISC_IV_WMI)
}

if("WISC_IV_PSI" %in% phenotype.filtered.data.column.names){
  phenotype.filtered.data$WISC_IV_PSI = as.integer(phenotype.filtered.data$WISC_IV_PSI)
}
