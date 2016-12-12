  # lasso_logistic_regression.R

  # This script constructs lasso logistic regression models on 1) fMRI fALFF data, 2) clinical/behavioral phenotype data, 3) fMRI fALFF + phenotype data. Hyperparameter tuning for
  # each model is accomplished through an exhaustive grid search over possible lambda (regularization) parameter values. The best model, by accuracy percentage on the training set,
  # is selected using 10-fold cross validation. A training/test split of 80% training, 20% test is used to perform the machine learning. After model construction, the script
  # constructs a confusion matrix for each model, which detail important model performance statistics, such as Accuracy %, Kappa statistic, and Sensitivity and Specificity.
  # Finally, an ROC plot, and AUC of ROC metrics are calculated. 

  # By Nipun Agarwala, Yuki Inoue, Axel Sly, Olver Bear Don't Walk and David Cohn

  set.seed(1)
  
  # Percentage of data allocated to training set for lasso logistic regression model construction
  training.percentage = 0.8
  
  # One hot encoded Phentoype Columns (by column index) to exclude from lasso logistic regression models

  phenotype.columns.exclude = c(1, 2, 3, 4, 16, 17, 18)
  
  lasso.phenotype.data = phenotype.encoded.data[,-phenotype.columns.exclude]
  
  # Conversion of autism Labels to type factor
  lasso.labels = rep("autism", length(phenotype.filtered.data$DX_GROUP))
  
  lasso.labels[as.integer(phenotype.filtered.data$DX_GROUP) == 2] = "control"
  
  lasso.labels = as.factor(lasso.labels)
  
  # Alpha value, which specifies application of lasso, L1 regularization penalty
  alpha = 1

  # Range of lambda regularization values considered in hyperparameter tuning
  lambda = seq(4, -2 , by = -1)

  lambda = 10^(lambda)

  # Creation of Hyperparameter Grid for Exhaustive Search
  lasso.logistic.regression.grid = expand.grid(alpha, lambda)

  names(lasso.logistic.regression.grid) = c("alpha", "lambda")

  # Use of 10-fold cross validation to select optimal hyperparameter values
  lasso.train.control = trainControl(method = "cv", number = 10, search = "grid")

  
  # Combine autism labels, fALFF and phenotype data
  
  fALFF.phenotype.data = cbind.data.frame(lasso.labels, fALFF, lasso.phenotype.data)
  
  # Create training (80% of data) and test sets (20% of data), using same proportion of autism vs.
  # control patients in each data set
  
  training.indices = createDataPartition(fALFF.phenotype.data$lasso.labels, times = 1, 
                                         p = training.percentage)
  
  training.indices = training.indices$Resample1
  
  # Training Data Set
  
  training.data = fALFF.phenotype.data[training.indices,]
  
  # Test Data Set
  
  test.data = fALFF.phenotype.data[-training.indices,]
  
  
  # fALFF Lasso Logistic Regresion Model
  fALFF.lasso.logistic.model = train(x = as.matrix(training.data[,2:ncol(fALFF)]), 
                             y = training.data$lasso.labels, family = "binomial", 
                             metric = "Accuracy", method = "glmnet", 
                             maximize = TRUE, trControl = lasso.train.control, 
                             tuneGrid = lasso.logistic.regression.grid)
  
  # fALFF predicted labels
  fALFF.pred.labels = predict(fALFF.lasso.logistic.model, as.matrix(test.data[,2:ncol(fALFF)]),
                              type = "raw")
  # fALFF confusion matrix
  fALFF.confusion.matrix = confusionMatrix(data = fALFF.pred.labels, 
                                           reference = test.data$lasso.labels)
  
  # phenotype Lasso Logistic Regression Model
  
  training.phenotype.data = training.data[,(ncol(fALFF) + 2) :ncol(training.data)]
  
  test.phenotype.data = test.data[,(ncol(fALFF) + 2) :ncol(test.data)]

  phenotype.lasso.logistic.model = train(x = training.phenotype.data, 
                                         y = training.data$lasso.labels, family = "binomial", 
                                         metric = "Accuracy", method = "glmnet", 
                                         maximize = TRUE, trControl = lasso.train.control, 
                                         tuneGrid = lasso.logistic.regression.grid)
  
  # phenotype predicted labels
  
  phenotype.pred.labels = predict(phenotype.lasso.logistic.model, test.phenotype.data,
                                  type = "raw")
  # phenotype confusion matrix
  
  phenotype.confusion.matrix = confusionMatrix(data = phenotype.pred.labels, 
                                               reference = test.data$lasso.labels)
  
  # fALFF + Phenotype Lasso Logistic Regression Model
  
  training.phenotype.fALFF.data = training.data[,-1]
  test.phenotype.fALFF.data = test.data[,-1]
  
  fALFF.phenotype.lasso.logistic.model = train(x = training.phenotype.fALFF.data,
                                               y = training.data$lasso.labels, family = "binomial",
                                               metric = "Accuracy", method = "glmnet",
                                               maximize = TRUE, trControl = lasso.train.control,
                                               tuneGrid = lasso.logistic.regression.grid)
  # fALFF + Phenotype predicted labels
  
  fALFF.phenotype.pred.labels = predict(fALFF.phenotype.lasso.logistic.model, test.phenotype.fALFF.data,
                                  type = "raw")
  
  # fALFF + Phenotype Confusion Matrix
  fALFF.phenotype.confusion.matrix = confusionMatrix(data = fALFF.phenotype.pred.labels, 
                                                     reference = test.data$lasso.labels)
  
  # ROC curves
  
  # fALFF predicted probabilities
  fALFF.pred.prob = predict(fALFF.lasso.logistic.model, as.matrix(test.data[,2:ncol(fALFF)]),
                            type = "prob")
  
  # fALFF ROC curve
  
  fALFF.roc.curve = pROC::roc(test.data$lasso.labels, fALFF.pred.prob[,2])
  
  plot(fALFF.roc.curve, main = "fALFF ROC Curve")
  
  # phenotype predicted probabilities
  
  phenotype.pred.prob = predict(phenotype.lasso.logistic.model, test.phenotype.data,
                                  type = "prob")
  
  # phenotype ROC curve
  
  phenotype.roc.curve = pROC::roc(test.data$lasso.labels, phenotype.pred.prob[,2])
  
  plot(phenotype.roc.curve, main = "Phenotype ROC Curve")
  
  # fALFF + phenotype predicted probabilities
  
  fALFF.phenotype.pred.prob = predict(fALFF.phenotype.lasso.logistic.model, test.phenotype.fALFF.data,
                                        type = "prob")
  
  # fALFF + phenotype ROC curve
  
  fALFF.phenotype.roc.curve = pROC::roc(test.data$lasso.labels, fALFF.phenotype.pred.prob[,2])
  
  plot(fALFF.phenotype.roc.curve, main = "fALFF and Phenotype ROC Curve")
  