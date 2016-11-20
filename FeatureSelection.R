# ensure the results are repeatable
set.seed(7)
# load the library
library(mlbench)
library(caret)
# load the data
trainData = read.csv("D:/Github/CSE_Data_Analysis/synthea_data_train_R.csv",header=TRUE)
x = trainData[,0:25]
y = trainData[,26]
# define the control using a random forest selection function
control <- rfeControl(functions=rfFuncs, method="repeatedcv", number=10)
# run the RFE algorithm
results <- rfe(x, y, sizes=c(1:8), rfeControl=control)
# summarize the results
print(results)
# list the chosen features
predictors(results)
# plot the results
plot(results, type=c("g", "o"))