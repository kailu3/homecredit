
# 1. Seperate data set into people who defaulted vs people who did not default

# Function to load the data in
load_application <- function(fileName) {
  read.csv(fileName, header = TRUE, sep = ",")
}

trainingData <- load_application("application_train.csv")

# Dealing with missing values in trainingData
trainingData$EXT_SOURCE_3[is.na(trainingData$EXT_SOURCE_3)] <- mean(trainingData$EXT_SOURCE_3, na.rm = T)
trainingData$EXT_SOURCE_2[is.na(trainingData$EXT_SOURCE_2)] <- mean(trainingData$EXT_SOURCE_2, na.rm = T)
trainingData$AMT_INCOME_TOTAL[is.na(trainingData$AMT_INCOME_TOTAL)] <- median(trainingData$AMT_INCOME_TOTAL, na.rm = T)
trainingData$AMT_CREDIT[is.na(trainingData$AMT_CREDIT)] <- median(trainingData$AMT_CREDIT, na.rm = T)
trainingData$DAYS_BIRTH[is.na(trainingData$DAYS_BIRTH)] <- mean(trainingData$DAYS_BIRTH, na.rm = T)
trainingData$AMT_ANNUITY[is.na(trainingData$AMT_ANNUITY)] <- median(trainingData$AMT_ANNUITY, na.rm = T)
trainingData$AMT_GOODS_PRICE[is.na(trainingData$AMT_GOODS_PRICE)] <- median(trainingData$AMT_GOODS_PRICE, na.rm = T)


splitData <- split(trainingData, trainingData$TARGET)

# LOAN DEFAULT -> failure to repay loan
default <- splitData[[2]]
nonDefault <- splitData[[1]]
(1- (propDefault <- length(default$TARGET) / (length(default$TARGET) + length(nonDefault$TARGET))))

# 2 Match it so that each category of defaultation has 50% of the sample's population
# random sampling function
randomSample = function(df,n) { 
  return (df[sample(nrow(df), n),])
}

smallerDF <- randomSample(nonDefault, 2.5 * length(default$TARGET))
fiftyFifty <- rbind(default, smallerDF)

# Get columns I want
slicedTraining <- fiftyFifty %>% 
  select(TARGET, FLAG_OWN_CAR, AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY,
         AMT_GOODS_PRICE, NAME_EDUCATION_TYPE,
         DAYS_BIRTH, DAYS_EMPLOYED,
         EXT_SOURCE_2, EXT_SOURCE_3, FLAG_WORK_PHONE, NAME_TYPE_SUITE, CNT_CHILDREN)


# 3. train model on entirity of slicedtraining data
training <- slicedTraining


# Logistic Regression Model
model <- glm(TARGET ~ DAYS_BIRTH + AMT_INCOME_TOTAL + AMT_CREDIT + EXT_SOURCE_2 +
               EXT_SOURCE_3 +
               AMT_ANNUITY + AMT_GOODS_PRICE + FLAG_OWN_CAR + NAME_EDUCATION_TYPE + 
               + FLAG_WORK_PHONE + DAYS_EMPLOYED,
             family = binomial(link='logit'), data = training)

# summary(model)
anova(model, test="Chisq")

# Predict on training set
set.seed(7)
sampleDF <- randomSample(trainingData, 100000)
predictions <- predict(model, sampleDF)

# Quick function to convert logit to probability
logit2prob <- function(logit){
  odds <- exp(logit)
  prob <- odds / (1 + odds)
  return(prob)
}

# Conversion
predictions <- logit2prob(predictions)

# Check our accuracy
fitted.results <- ifelse(predictions > 0.5, 1, 0)
misClasificError <- mean(fitted.results != sampleDF$TARGET)
print(paste('Accuracy', 1-misClasificError))

(confusion <- table(fitted.results, sampleDF$TARGET))
# 4. Build the model around the training data

# 5. Test the model on the test data and TEST how good it is.
testData$EXT_SOURCE_1[is.na(testData$EXT_SOURCE_1)] <- mean(testData$EXT_SOURCE_1, na.rm = T)
testData$EXT_SOURCE_2[is.na(testData$EXT_SOURCE_2)] <- mean(testData$EXT_SOURCE_2, na.rm = T)
testData$EXT_SOURCE_3[is.na(testData$EXT_SOURCE_3)] <- mean(testData$EXT_SOURCE_3, na.rm = T)
testData$AMT_INCOME_TOTAL[is.na(testData$AMT_INCOME_TOTAL)] <- median(testData$AMT_INCOME_TOTAL, na.rm = T)
testData$AMT_CREDIT[is.na(testData$AMT_CREDIT)] <- median(testData$AMT_CREDIT, na.rm = T)
testData$DAYS_BIRTH[is.na(testData$DAYS_BIRTH)] <- mean(testData$DAYS_BIRTH, na.rm = T)
testData$AMT_ANNUITY[is.na(testData$AMT_ANNUITY)] <- median(testData$AMT_ANNUITY, na.rm = T)
testData$AMT_GOODS_PRICE[is.na(testData$AMT_GOODS_PRICE)] <- median(testData$AMT_GOODS_PRICE, na.rm = T)

new.predictions <- predict(model, testData)
new.predictions <- logit2prob(new.predictions)
new.predictions

# Convert to df and edit format
predictions.df <- as.data.frame(new.predictions)
test.df <- as.data.frame(testData$SK_ID_CURR)
final <- cbind(test.df, predictions.df)
final <- rename(final, SK_ID_CURR = "testData$SK_ID_CURR")
final <- rename(final, TARGET = "new.predictions")

# Save as file
write.csv(final,'logisticReg2.csv')
