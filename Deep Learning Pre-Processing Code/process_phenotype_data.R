  # process_phenotype_data.R

  # This script is responsible for processing fALFF clinical and behavioral phenotype data 
  # associated with the ABIDE data set. This script: 1) removes features that have greater
  # than a certain threshold of its data missing (missing.data.factor), 2) designates the remaining 
  # missing values with the encoding -9999, 3) performs random forest based multiple imputation that
  # can robustly impute both categorical and numeric features, 4) one hot encodes the remaining
  # categorical variables.

  # By Nipun Agarwala, Yuki Inoue, Axel Sly, Olver Bear Don't Walk and David Cohn

  # Converting factor data types to strings
  phenotype.data = phenotype_data %>%
    mutate_if(is.factor, as.character)

  # Converting empty strings to null values (NAs)
  remove_empty_strings = function(x){
    ifelse(as.character(x) != "", x, NA)
  }
  
  phenotype.data = phenotype.data %>%
    mutate_each(funs(remove_empty_strings))

  # Maximum percentage (i.e. 0.25 -> 25%) of missing data allowed in data filtration step
  missing.data.factor = 1.00

  # Calculates number of null entries associated with each phenotypic feature
  source(paste(code.directory, 'calculate_null_values.R', sep = "/"))

  # Excluding feature if maximum percentage of missing data entries is exceeded

  if(nrow(excluded.features) > 0) {
    features.to.exclude=rep("a", times = nrow(excluded.features))
    for (i in 1:nrow(excluded.features)) {
      features.to.exclude[i] = (excluded.features)[i,1]
    }
    phenotype.filtered.data = phenotype.data[,-which(names(phenotype.data) %in% features.to.exclude)]
  } else {
    phenotype.filtered.data = phenotype.data
  }

  # Designate remaining nulls as -9999
  change_nulls = function(x) {
    if((sapply(x, class))[1] == "character"){
      ifelse(is.na(x), "-9999", x)
    } else if((sapply(x, class))[1] == "numeric") {
      ifelse(is.na(x), -9999.0, x)
    } else if((sapply(x, class))[1] == "integer"){
      ifelse(is.na(x), -9999, x)
    }
  }

  phenotype.filtered.data = phenotype.filtered.data %>%
    mutate_each(funs(change_nulls))

  # Perform random-forest based multiple imputation
  source(paste(code.directory, 'multiple_imputation.R', sep = "/"))

  # Change remaining applicable categorical columns to one-hot encoding 
  source(paste(code.directory, 'apply_one_hot_encoding.R', sep = "/"))

  # Output processed clinical phenotype data to a .csv file
  write.csv(phenotype.encoded.data, 'processed_phenotype_data.csv')