# Machine Learning: Human Activity Performance Evaluation
Paul Clark  
April 15, 2017  



# Introduction

### Background

Using wearable devices like Fitbit, much data on personal activity can be collected cheaply. The goal of this exercise is to use data from motion sensors on the belt, forearm, arm, and dumbbell of 6 participants to recognize the manner in which they performed dumbbell lifts: whether they did a specified lift that was the same for all trials correctly ("A"), or in one of 4 specified incorrect ways ("B" through "D"). More information and data was available here as of 4/18/17: <<http://groupware.les.inf.puc-rio.br/har>>. See the section _"Weight Lifting Exercise Dataset"_ [@Velloso2013].  Note that all information here that is drawn from the original study is licensed under Creative Commons (CC BY-SA): it can be used for any purpose as long as the original authors are cited.

### Objectives
  
  *  Given training data extracted from the original study, build a model to recognize, based on sensor data, the manner (`classe`) in which each of the 6 participants did any given barbell lift
  * In particular, predict the values (A through D) of 20 unlabeled observations in a testing dataset.

### Report

This report describes key steps in the analysis, including rationale for key choices.  Featured elements include:

  * Model building
  * Model validation
  * Expected out-of-sample error

### Conclusions

A random forest model provides high prediction accuracy for "held out" observations of subjects who participate in model training (~ 99%), but low accuracy on completely new participants (~ 36%).  Note, however, that due to lengthy computing times, we use the default model parameters, neglecting model tuning.  From the literature, e.g. @elements, we guess that model tuning of parameter `mtry` (the number of variables evaluated for each split in the trees) might enhance performance in the 'new participant' case.  Another option to investigate would be basing prediction on summary features used by the orginal authors, but not used in the test data for this assignment. 

# Preparation

### Data Retrieval

Data was obtained via the links below on 4/16/17. We load it using package `readr`.  

```r
if (!file.exists("training.csv")){
        download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", "training.csv")
}
if (!file.exists("testing.csv")){
        download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", "testing.csv")
}
if (!"readr" %in% rownames(installed.packages())) install.packages("readr")
library(readr)

colspec<-list(user_name = col_factor(levels = c("adelmo","carlitos","charles","eurico","jeremy","pedro")), 
                classe = col_factor(levels = c("A","B","C","D","E")))

# guess_max set at 6000 to capture decimal values first occurring after row 1000
training <- read_csv("training.csv", na = c("","NA","#DIV/0!"), col_types = colspec, guess_max = 6000)
testing <- read_csv("testing.csv", na = c("","NA","#DIV/0!"), col_types = colspec)
```
### Data Pre-Processing

Summary of the data via `str()` is omitted here due to length. The training data has 160 predictors and 19622 observations.

```r
str(training, list.len = ncol(training), give.attr = FALSE)
str(testing, list.len = ncol(testing), give.attr = FALSE)
```
In the original analysis, raw sensor signals were sampled at 45 Hz over windows of time from 0.5 to 2.5 seconds. Summary "features" (variable names in the dataset beginning with "`avg_`", "`stddev_`", "`kurtosis_`", "`skewness_`", etc.) were computed in post-processing of signals over each window. Both summary features and 45 Hz signals are provided in the training data, but the summary features are not provided in the test data.  Therefore, the summary features are not useful for prediction on the test set and are excluded from the modelling. Additionally, no rows with `new_window == 'yes'`, which contain the summary features, appear in the test set, so these are also excluded. Finally, in the absence of contextual documentation, we see no way to make use of the "`_timestamp_`" or "`_window`" information in predicting test labels, therefore we exclude it, too.  We use package `dplyr` to select only the remaining information.

```r
if (!"dplyr" %in% rownames(installed.packages())) install.packages("dplyr"); library(dplyr)

training <- filter(training,new_window!="yes")%>%select(-c(X1,raw_timestamp_part_1:num_window,starts_with(
        "avg"),starts_with("stddev"),starts_with("kurt"),starts_with("skew"),starts_with("max"),starts_with(
        "min"),starts_with("amplitude"),starts_with("var")))

testing<-select(testing,-c(X1,raw_timestamp_part_1:num_window,starts_with("avg"),starts_with("stddev"),
        starts_with("kurt"),starts_with("skew"),starts_with("max"),starts_with("min"),starts_with("amplitude"),
        starts_with("var")))
```

After the exclusions, we are left with 53  potential predictors.  

```r
names(training)[-length(training)]
```

```
##  [1] "user_name"            "roll_belt"            "pitch_belt"           "yaw_belt"            
##  [5] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"         "gyros_belt_z"        
##  [9] "accel_belt_x"         "accel_belt_y"         "accel_belt_z"         "magnet_belt_x"       
## [13] "magnet_belt_y"        "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [17] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"          "gyros_arm_y"         
## [21] "gyros_arm_z"          "accel_arm_x"          "accel_arm_y"          "accel_arm_z"         
## [25] "magnet_arm_x"         "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [29] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell" "gyros_dumbbell_x"    
## [33] "gyros_dumbbell_y"     "gyros_dumbbell_z"     "accel_dumbbell_x"     "accel_dumbbell_y"    
## [37] "accel_dumbbell_z"     "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [41] "roll_forearm"         "pitch_forearm"        "yaw_forearm"          "total_accel_forearm" 
## [45] "gyros_forearm_x"      "gyros_forearm_y"      "gyros_forearm_z"      "accel_forearm_x"     
## [49] "accel_forearm_y"      "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [53] "magnet_forearm_z"
```

# Cross-Validation Strategy

Based on the training data, we create a validation partition with which to assess performance.  The `caret` package produces a partition stratified with respect to the 5 `classe` outcomes. Note that `caret` requires the `lattice` and `ggplot2` packages, by default.

```r
if (!"caret" %in% rownames(installed.packages())) install.packages("caret")
if (!"lattice" %in% rownames(installed.packages())) install.packages("lattice")
if (!"ggplot2" %in% rownames(installed.packages())) install.packages("ggplot2")
library(caret)

set.seed(1)
inTrain <- createDataPartition(y = training$classe, p = 0.7, list = FALSE)
Train <- training[inTrain,]
Validation <- training[-inTrain,]
```
  
  
# Model Building and Validation
  
### Prediction modeling for "in-sample" participants

Following the authors of the original paper, we attempt prediction based on random forests.  Note that the formula interface for `randomForest()` is not used, as it generates high overhead when there are many variables, and that minimum `nodesize` is set to 10 (the default is 1).  These steps were taken to make run times feasible.  Below, the cumulative out-of-bag error is shown for every 25 trees, indicating that model performance is adequate by 500 (the default). 

```r
if (!"randomForest" %in% rownames(installed.packages())) install.packages("randomForest")
library(randomForest)
set.seed(921)
rfMod <- randomForest(x=Train[,-54], y=Train$classe, proximity = FALSE, nodesize = 10, do.trace = 25)
```

```
## ntree      OOB      1      2      3      4      5
##    25:   1.72%  0.55%  2.84%  2.09%  2.68%  1.13%
##    50:   1.21%  0.37%  1.81%  2.05%  1.86%  0.53%
##    75:   1.03%  0.26%  1.81%  1.49%  1.72%  0.32%
##   100:   0.91%  0.21%  1.46%  1.41%  1.59%  0.36%
##   125:   0.85%  0.16%  1.34%  1.32%  1.59%  0.32%
##   150:   0.79%  0.10%  1.15%  1.28%  1.59%  0.28%
##   175:   0.77%  0.13%  1.15%  0.94%  1.72%  0.32%
##   200:   0.73%  0.10%  0.96%  1.07%  1.68%  0.28%
##   225:   0.73%  0.13%  1.00%  1.15%  1.54%  0.24%
##   250:   0.71%  0.13%  0.92%  1.07%  1.59%  0.24%
##   275:   0.70%  0.10%  0.92%  1.07%  1.54%  0.28%
##   300:   0.70%  0.13%  0.96%  1.07%  1.45%  0.28%
##   325:   0.71%  0.16%  0.96%  1.02%  1.50%  0.28%
##   350:   0.71%  0.16%  0.88%  1.11%  1.50%  0.28%
##   375:   0.69%  0.13%  0.85%  1.07%  1.50%  0.32%
##   400:   0.71%  0.10%  0.92%  1.07%  1.50%  0.36%
##   425:   0.74%  0.10%  0.96%  1.19%  1.50%  0.36%
##   450:   0.74%  0.13%  0.96%  1.15%  1.54%  0.32%
##   475:   0.74%  0.13%  0.96%  1.15%  1.50%  0.36%
##   500:   0.69%  0.08%  0.88%  1.07%  1.50%  0.36%
```

We next compute the confusion matrix and associated statistics on the validation set.  Performance appears quite good. Note that we find it necessary to reference `predict` by its package name - it doesn't appear to be directly available in the namespace, at least with the current collection of packages loaded.


```r
rfPred <- randomForest:::predict.randomForest(rfMod, newdata = Validation[,-54], type = "response")
rfConf <- confusionMatrix(rfPred, Validation$classe)
rfConf
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1639   11    0    0    0
##          B    2 1102    9    0    0
##          C    0    2  996   24    1
##          D    0    0    0  920    5
##          E    0    0    0    0 1052
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9906         
##                  95% CI : (0.9878, 0.993)
##     No Information Rate : 0.2847         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9881         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9988   0.9883   0.9910   0.9746   0.9943
## Specificity            0.9973   0.9976   0.9943   0.9990   1.0000
## Pos Pred Value         0.9933   0.9901   0.9736   0.9946   1.0000
## Neg Pred Value         0.9995   0.9972   0.9981   0.9950   0.9987
## Prevalence             0.2847   0.1935   0.1744   0.1638   0.1836
## Detection Rate         0.2844   0.1912   0.1728   0.1596   0.1825
## Detection Prevalence   0.2863   0.1931   0.1775   0.1605   0.1825
## Balanced Accuracy      0.9981   0.9930   0.9927   0.9868   0.9972
```

Finally, we compute the predicted values for the unlabeled testing data, which are kept hidden due to the plagiarism policy.


```r
randomForest:::predict.randomForest(rfMod, newdata = testing, type = "response")
```

#### Out-of-sample error for "in-sample" participants

Based on the above, we expect an out-of-sample error rate of **0.94%**.  Note that this error rate is slightly higher than the 'out-of-bag' estimate of 0.69% from model training.  This is probably just a statistical fluctuation: the difference is inside the 95% confidence interval provided in the confusion matrix results.

### Prediction modeling for "out-of-sample" participants

Before closing, we use cross-validation to investigate the model's accuracy on participants for which there is no training data.  Performance is rather poor - the model fails to generalize to unknown participants. This seems a major limitation of this approach.  We could try to 'tune' some of the parameters to improve performance (in particular, `mtry`) but this would probably only generate marginal improvements, and compute-time too excessive to perform the required iterations.  Note: the original authors achieve accuracies on order ~ 75%, but they use the processed data we have discarded, not the raw signals we were forced to use by the structure of the test data.

```r
parts <- c("adelmo","carlitos","charles","eurico","jeremy","pedro")

rfPredLOPO_Tot <- NULL
partTest_Tot <- NULL
set.seed(309)
for (i in 1:length(parts)){
        partTest <- filter(training, user_name==parts[i]) %>% select(-user_name)
        partTrain <- filter(training, user_name!=parts[i]) %>% select(-user_name)
        rfModLOO<-randomForest(x=select(partTrain,-classe),y=partTrain$classe,prox=FALSE,nodesize=10,do.trace= 20)
        rfPredLOPO <- randomForest:::predict.randomForest(rfModLOO, newdata=partTest, type = "response")
        partTest_Tot <- rbind(partTest_Tot, partTest)
        rfPredLOPO_Tot <- c(rfPredLOPO_Tot,rfPredLOPO)
}

# Convert classe labels back to A to E factors: they become 1 to 5 during concatenation in `for` loop
classelabels = c("A","B","C","D","E")
rfPredLOPO_Tot <- factor(classelabels[rfPredLOPO_Tot], levels = classelabels)
```
  

We show the confusion matrix and statistics for leave-one-participant-out cross-validation:

```r
(rfLOPOConf <- confusionMatrix(rfPredLOPO_Tot,partTest_Tot$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2494  576  628  679   99
##          B  483 1493  839 1028  543
##          C    0   21  271   12   15
##          D    2   38  138  124  163
##          E 2492 1590 1476 1304 2708
## 
## Overall Statistics
##                                           
##                Accuracy : 0.369           
##                  95% CI : (0.3621, 0.3758)
##     No Information Rate : 0.2847          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.2025          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.4559   0.4016  0.08085 0.039403   0.7676
## Specificity            0.8558   0.8133  0.99697 0.978779   0.5626
## Pos Pred Value         0.5572   0.3404  0.84953 0.266667   0.2830
## Neg Pred Value         0.7980   0.8500  0.83696 0.838782   0.9150
## Prevalence             0.2847   0.1935  0.17444 0.163770   0.1836
## Detection Rate         0.1298   0.0777  0.01410 0.006453   0.1409
## Detection Prevalence   0.2329   0.2282  0.01660 0.024199   0.4980
## Balanced Accuracy      0.6558   0.6074  0.53891 0.509091   0.6651
```


# Conclusions

The methods above provide high recognition performance for "in-sample" participants (overall error rate of **0.94%**).  However, a training period per participant is implied.  With overall error rate of **63.1%**, the model does not generalize well to "out-of-sample" participants.




# Bibliography
  
  
