  # calculate_null_values.R

  # This script calculates the number of missing values associated with each feature in the clinical
  # phenotype data from the ABIDE data set. 

  # By Nipun Agarwala, Yuki Inoue, Axel Sly, Olver Bear Don't Walk and David Cohn


  # Convert data from wide format (samples along rows, features along columns) to long format
  # (stacking patient + feature combinations data) in order to identify the number of missing values
  # associated with each feature type

  phenotype.data.long = gather(phenotype.data, feature, feature.value, -SITE_ID, -SUB_ID)

  # Identify Features with no missing values 

  non.null.features=phenotype.data.long %>%
    filter(!is.na(feature.value)) %>%
    group_by(feature) %>%
    summarise(non.nulls = n()) %>%
    filter(non.nulls == nrow(phenotype.data))

  # Vector storing number of non-missing values associated with each feature
  non.null.values=rep(0, times = nrow(non.null.features))

  # Dataframe storing feature names that are not missing any values, as well as number of 
  # non-missing feature values
  non.null.features.values.dataframe = data.frame(non.null.features$feature, non.null.values)

  colnames(non.null.features.values.dataframe) = c("feature", "number.nulls")

  # List of Features by Number of Missing Values

  number.null.features = phenotype.data.long %>%
    filter(is.na(feature.value)) %>%
    group_by(feature) %>%
    summarise(number.nulls = n()) %>%
    arrange(desc(number.nulls))

  number.null.features = rbind.data.frame(number.null.features, 
                                          non.null.features.values.dataframe)
  
  # Features that will be excluded because the percentage of missing values exceeds the maximum
  # threshold
  excluded.features = number.null.features %>%
    filter(number.nulls > missing.data.factor * nrow(phenotype.data)) %>%
    dplyr::select(feature)
