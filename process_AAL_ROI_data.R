  # process_AAL_ROI_data.R

  # This script is responsible for processing AAL ROI data. This script combines AAL ROI data
  # with region name information.

  # By Nipun Agarwala, Yuki Inoue, Axel Sly, Olver Bear Don't Walk and David Cohn

  # For loop helps process AAL_ROI for each patient
  for (i in 1:length(AAL_ROI)){
    # patient specific AAL_ROI data
    AAL.data = (AAL_ROI[[i]])
    colnames(AAL.data) = region_name$V3
    write.csv(AAL.data, paste(names(AAL_ROI[i]),'_ROI_data.csv', sep = ""), 
             row.names = TRUE, col.names = TRUE)
  }