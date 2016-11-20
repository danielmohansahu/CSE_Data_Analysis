# load the library
library(mlbench)
library(caret)
# load the dataset
trainData = read.csv("D:/Github/CSE_Data_Analysis/synthea_data_train_R.csv",header=TRUE)
testData = read.csv("D:/Github/CSE_Data_Analysis/synthea_data_test_R.csv", header = TRUE)
x = trainData[,0:25]
y = trainData[,26]
# prepare training scheme
control = trainControl(method="repeatedcv", number=10, repeats=3)
# train the LVQ model
set.seed(7)
modelLvq <- train(Heart.Disease~., data=trainData, method="lvq", trControl=control)
# train the GBM model
set.seed(7)
modelGbm <- train(Heart.Disease~., data=trainData, method="gbm", trControl=control, verbose=FALSE)
# train the SVM model
set.seed(7)
modelSvm <- train(Heart.Disease~., data=trainData, method="svmRadial", trControl=control)
# train the RF model
set.seed(7)
modelRF <- train(Heart.Disease~., data=trainData, method="rf", trControl=control)
# collect resamples
results <- resamples(list(LVQ=modelLvq, GBM=modelGbm, SVM=modelSvm, RF=modelRF))
# summarize the distributions
summary(results)
# boxplots of results
bwplot(results)
# dot plots of results
dotplot(results)