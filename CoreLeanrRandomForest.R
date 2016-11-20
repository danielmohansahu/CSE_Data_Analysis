library(randomForest)
library(mlbench)
library(caret)
library(pROC)
library(CORElearn)

cat(versionCore(),"\n")
# load data
trainData = read.csv("D:/Github/CSE_Data_Analysis/synthea_data_train_R.csv",header=TRUE)
testData = read.csv("D:/Github/CSE_Data_Analysis/synthea_data_test_R.csv", header = TRUE)
#x = trainData[,0:25]
#y = trainData[,26]

# build random forests model with certain parameters
# setting maxThreads to 0 or more than 1 forces utilization of several processor cores
modelRF <- CoreModel(Heart.Disease~. , trainData, model="rf",
                     selectionEstimator="InfGain",minNodeWeightRF=10,
                     rfNoTrees=1000, maxThreads=10)
print(modelRF) # simple visualization, test also others with function plot

pred <- predict(modelRF, testData, type="both") # prediction on testing set
print (pred)
#evaluating model
modelEval(model = modelRF, correctClass = testData$Heart.Disease, predictedClass = pred$class)
require(ExplainPrediction)


#not run this yet... take couple hours to generate a graph... dont think worth it.
#explainVis(modelRF, train, test, method="EXPLAIN",visLevel="model",
          # problemName="data", fileType="none", classValue=1, displayColor="color")
