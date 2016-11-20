library(randomForest)
library(mlbench)
library(caret)
library(gbm)
library(dplyr)

trainData = read.csv("D:/Github/CSE_Data_Analysis/CSE_Data_Analysis/synthea_data_train.csv",header=TRUE)
testData = read.csv("D:/Github/CSE_Data_Analysis/CSE_Data_Analysis/synthea_data_test.csv", header = TRUE)
x = trainData[,1:24]
y = trainData[,25]
summary(trainData)


fit = gbm.fit(x, y, distribution="bernoulli", n.trees = 7000, shrinkage = 0.001, interaction.depth = 3, n.minobsinnode = 10)

# summarize the fit
summary(fit)
# make predictions
predictions = predict(fit, testData)
# summarize accuracy
rmse = mean((testData$Heart.Disease - predictions)^2)
print(rmse)