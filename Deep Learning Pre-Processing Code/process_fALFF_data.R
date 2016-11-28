  # process_fALFF_data.R

  # This script is responsible for processing fALFF fMRI imaging data. This script combines
  # voxel data with region coordinate and region description data. 

  # By Nipun Agarwala, Yuki Inoue, Axel Sly, Olver Bear Don't Walk and David Cohn

  colnames(region_name) =c("order", "region.id", "region.name")

  # For loop helps extract fALFF data for each patient
  for (k in 1:nrow(fALFF)){
    # patient specific fMRI fALFF data
    patient.ALFF = fALFF[k,]
    # combine region code and region coordinate dataframes
    region.code.coord = cbind(region_code, coord)
    # combine patient fALFF data with region code/coordinate infomration
    patient.fALFF.data = cbind(patient.fALFF, region.code.coord)
    colnames(patient.fALFF.data) = c("voxel.value", "region.code", "x", "y", "z")
    patient.fALFF.data = as.data.frame(patient.fALFF.data)
    # combine fALFF + region code + coordinate data with region descriptions
    patient.fALFF.data = left_join(patient.fALFF.data, region_name, 
                                   by = c("region.code"="region.id"))
    # Extract Subject ID from filtered phenotype dataframe
    patient.information = phenotype.filtered.data[k,]
    write.csv(patient.fALFF.data, paste(patient.information[2], '.csv', sep = ""))
}