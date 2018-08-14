# =====================================================================================================================
# =                                                                                                                   =
# =                                                                                                                   =
# = Author: Kai Lu <kailu3@sas.upenn.edu> | @kailu3                                                               =
# =====================================================================================================================
# Classifying Loans: Default Rate Prediction

# LIBRARIES ------------------------------------------------------------------------------------------------------------
library(dplyr)
library(purrr)
library(caret)
library(readr)

# METHOD ---------------------------------------------------------------------------------------------------------------

cat("Reading Data....\n")
train <- read.csv("trainingData.csv")
test <- read.csv("testData.csv")


y <- factor(ifelse(train$TARGET == 1, "Y", "N"))
train$TARGET <- NULL
df.temp <- rbind(train, test) 

# Change all columns into numeric
df.temp <- df.temp %>% mutate_if(is.factor, as.numeric)

# Do imputation with Median
df.temp <- map_df(df.temp, function(x) {
  x[is.na(x)] <- median(x, na.rm = TRUE); x })

# convert tibble to df
df.temp <- df.temp %>% as.data.frame()


# Split back to train/test
n <- nrow(train)
trainingData <- df.temp[1:n, ]
testData <- df.temp[(n + 1):nrow(df.temp), ]

# Preparing our testing data set for model building
cat("Removing low variance items....\n")
nz <- nearZeroVar(trainingData, freqCut = 2000, uniqueCut = 10)
trainingData <- trainingData[,-nz]

cat("Removing highly correlated features.... \n")
df.correlated <- findCorrelation(cor(trainingData), cutoff = 0.65, verbose = TRUE, exact = TRUE)
trainingData <- trainingData[, -df.correlated]


## Perform grid search with caret
cat("Training our model.... \n")
trControl <- trainControl(method = "cv", n = 5, classProbs = TRUE, summaryFunction = twoClassSummary,
                          savePredictions = 'final', verboseIter = TRUE)
grid <- expand.grid(nrounds = c(150), max_depth = c(8), eta = 0.05, 
                    gamma = 0, colsample_bytree = 0.8, min_child_weight = 0, subsample = 0.8)

# preProcValues <- preProcess(trainingData, method = c("center", "scale"))

mod_xgb <- train(trainingData[, -1], y, method = "xgbTree", metric = "ROC", tuneGrid = grid, 
                 trControl = trControl,  preProcess = NULL)
# Check model
mod_xgb

# Generate predictions on testData
pred <- predict(mod_xgb, testData, type = "prob")$Y
sub <- data.frame(SK_ID_CURR = testData$SK_ID_CURR,TARGET=pred)
write.csv(sub,'xgbTree.csv',row.names = F)
