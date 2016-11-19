library(randomForest)
library(mlbench)
library(caret)
library(gbm)

trainData = read.csv("D:/Github/CSE_Data_Analysis/synthea_data_train_R.csv",header=TRUE)
testData = read.csv("D:/Github/CSE_Data_Analysis/synthea_data_test_R.csv",header=TRUE)

fit = gbm(Heart.Disease~., data=trainData, distribution="gaussian")

# summarize the fit
summary(fit)
# make predictions
predictions = predict(fit, testData)
# summarize accuracy
rmse = mean((testData$Heart.Disease - predictions)^2)
print(rmse)