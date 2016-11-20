library(randomForest)
library(mlbench)
library(caret)
library (e1071)

# Load Dataset
trainData = read.csv("D:/Github/CSE_Data_Analysis/synthea_data_train_R.csv",header=TRUE)
testData = read.csv("D:/Github/CSE_Data_Analysis/synthea_data_test_R.csv", header = TRUE)
x = trainData[,0:25]
y = trainData[,26]


# Create model with default paramters
control <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 7
metric <- "Accuracy"
set.seed(seed)
mtry <- sqrt(ncol(x))
tunegrid <- expand.grid(.mtry=mtry)
rf_default <- train(x, y, method="rf", tuneGrid=tunegrid, trControl=control)
print(rf_default)

#testing

# Random Search
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="random")
set.seed(seed)
mtry <- sqrt(ncol(x))
rf_random <- train(x, y, method="rf", metric=metric, tuneLength=25, trControl=control)
print(rf_random)
plot(rf_random)

#Grid Search
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
set.seed(seed)
tunegrid <- expand.grid(.mtry=c(1:25))
rf_gridsearch <- train(x, y, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_gridsearch)
plot(rf_gridsearch)

#-----------------------------------------------------------------------------------------------------------
#ROC
# Create model with default paramters
control <- trainControl(method="repeatedcv", number=10, repeats=3, classProbs = TRUE, summaryFunction = twoClassSummary)
seed <- 7
metric <- "ROC"
set.seed(seed)
mtry <- sqrt(ncol(x))
tunegrid <- expand.grid(.mtry=mtry)
rf_default <- train(x, y, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_default)

#testing

# Random Search
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="random", classProbs = TRUE, summaryFunction = twoClassSummary)
set.seed(seed)
mtry <- sqrt(ncol(x))
rf_random <- train(x, y, method="rf", metric=metric, tuneLength=25, trControl=control)
print(rf_random)
plot(rf_random)

#Grid Search
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid",classProbs = TRUE, summaryFunction = twoClassSummary)
set.seed(seed)
tunegrid <- expand.grid(.mtry=c(1:25))
rf_gridsearch <- train(x, y, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_gridsearch)
plot(rf_gridsearch)