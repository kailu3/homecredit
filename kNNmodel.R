# =====================================================================================================================
# =                                                                                                                   =
# =                                                                                                                   =
# = Author: Kai Lu <kailu3@sas.upenn.edu> | @kailu3                                                               =
# =====================================================================================================================

# Classifying Loans
# TARGET

# LIBRARIES ------------------------------------------------------------------------------------------------------------

library(dplyr)
library(ggplot2)
library(readr)
library(readxl)
library(kknn)

# METHOD ---------------------------------------------------------------------------------------------------------------

cat("Loading data in... \n")

# Read Test Data in
test = "application_test.csv"
testData <- read.csv(test, header = TRUE, sep = ",")
testData[is.na(testData)] <- 0
head(testData)

# Read Training Data in
training = "application_train.csv"
trainingData <- read.csv(training, header = TRUE, sep = ",")
trainingData[is.na(trainingData)] <- 0


colnames(trainingData)

# Use a kknn model to fit most significant columns
training.kknn <- kknn(TARGET ~ DAYS_BIRTH + EXT_SOURCE_3 + EXT_SOURCE_2 + EXT_SOURCE_1
                      + AMT_CREDIT,
                      trainingData, testData, k = 10, kernel = "rectangular", distance = 2)
predictions <- predict(training.kknn)
length(predictions)
predictions <- as.data.frame(predictions)

# create target column
colnames(predictions)
predictions <- rename(predictions, TARGET = "predictions") 

# extract SK IDs
ids <- testData[, 1]
ids <- as.data.frame(ids)

# bind the two columns
file <- cbind(ids, predictions)
file <- rename(file, SK_ID_CURR = "ids")
write.csv(file,'kaggleFirstSub.csv')
