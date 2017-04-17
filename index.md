# Machine Learning: Human Activity Performance Evaluation
Paul Clark  
April 15, 2017  



# Introduction

### Background

Using wearable devices like Fitbit, large amounts of data on personal activity can be collected  inexpensively. The goal of this exercise is to use data from motion sensors on the belt, forearm, arm, and dumbbell of 6 participants to reliably recognize the manner in which they performed dumbbell lifts: specifically, whether they did a particular sort of lift correctly ("A"), or did it in one of 4 incorrect ways ("B" through "D"). More information and data is available here: <http://groupware.les.inf.puc-rio.br/har>. See the section on the _**Weight Lifting Exercise Dataset**_, [@Velloso2013].  All the data here is licensed under Creative Commons (CC BY-SA): it can be used for any purpose as long as the original authors are cited.

### Objectives
  
  *  Given training data extracted from the original study, build a model to recognize, based on sensor data, the manner (`classe`) in which each of the 6 participants did any given barbell lift
  * In particular, predict the values (A through D) of 20 unlabeled observations in a testing dataset.

### Report

This report describes key steps in the analysis, including rationale for key choices.  Featured elements include:

  * Model building
  * Model validation
  * Expected out-of-sample error
     
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

colspec <- list(user_name = col_factor(levels = c("adelmo","carlitos","charles","eurico","jeremy","pedro")), 
                classe = col_factor(levels = c("A","B","C","D","E")))

# guess_max set at 6000 to capture decimal values first occurring after row 1000
training <- read_csv("training.csv", na = c("","NA","#DIV/0!"), col_types = colspec, guess_max = 6000)
testing <- read_csv("testing.csv", na = c("","NA","#DIV/0!"), col_types = colspec)
```
### Data Pre-processing

We use the `str()` function to examine the data. The summary is omitted here due to length: the training data has 160 predictors and 19622 observations.

```r
str(training, list.len = ncol(training), give.attr = FALSE)
str(testing, list.len = ncol(testing), give.attr = FALSE)
```
In the original analysis, raw sensor signals were sampled at 45 Hz over windows of time from 0.5 to 2.5 seconds. Summary "features" (variable names in the dataset beginning with "`avg_`", "`stddev_`", "`kurtosis_`", "`skewness_`", etc.) were computed in post-processing of signals over each window. Both features and signals are provided in the training data, but the features are not provided in the test data.  Therefore, the features are not useful for prediction on the test set and are excluded from the modelling. Additionally, no rows with `new_window == 'yes'`, which contain the computed features, appear in the test set, so these are also excluded. Finally, in the absence of contextual documentation, we see no way to make use of the "`_timestamp_`" or "`_window`" information in predicting test labels, therefore we exclude it, too.  We use package `dplyr` to filter out the unneeded information.

```r
if (!"dplyr" %in% rownames(installed.packages())) install.packages("dplyr")
library(dplyr)

training <- filter(training, new_window != "yes") %>%
        select(-c(X1, raw_timestamp_part_1:num_window, starts_with("avg"),
               starts_with("stddev"),starts_with("kurt"),starts_with("skew"),starts_with("max"),
               starts_with("min"),starts_with("amplitude"),starts_with("var")))

testing <- select(testing, -c(X1, raw_timestamp_part_1:num_window, starts_with("avg"),
               starts_with("stddev"),starts_with("kurt"),starts_with("skew"),starts_with("max"),
               starts_with("min"),starts_with("amplitude"),starts_with("var")))
```

After the exclusions, we are left with 53  potential predictors.  

```r
names(training)[-length(training)]
```

```
##  [1] "user_name"            "roll_belt"            "pitch_belt"          
##  [4] "yaw_belt"             "total_accel_belt"     "gyros_belt_x"        
##  [7] "gyros_belt_y"         "gyros_belt_z"         "accel_belt_x"        
## [10] "accel_belt_y"         "accel_belt_z"         "magnet_belt_x"       
## [13] "magnet_belt_y"        "magnet_belt_z"        "roll_arm"            
## [16] "pitch_arm"            "yaw_arm"              "total_accel_arm"     
## [19] "gyros_arm_x"          "gyros_arm_y"          "gyros_arm_z"         
## [22] "accel_arm_x"          "accel_arm_y"          "accel_arm_z"         
## [25] "magnet_arm_x"         "magnet_arm_y"         "magnet_arm_z"        
## [28] "roll_dumbbell"        "pitch_dumbbell"       "yaw_dumbbell"        
## [31] "total_accel_dumbbell" "gyros_dumbbell_x"     "gyros_dumbbell_y"    
## [34] "gyros_dumbbell_z"     "accel_dumbbell_x"     "accel_dumbbell_y"    
## [37] "accel_dumbbell_z"     "magnet_dumbbell_x"    "magnet_dumbbell_y"   
## [40] "magnet_dumbbell_z"    "roll_forearm"         "pitch_forearm"       
## [43] "yaw_forearm"          "total_accel_forearm"  "gyros_forearm_x"     
## [46] "gyros_forearm_y"      "gyros_forearm_z"      "accel_forearm_x"     
## [49] "accel_forearm_y"      "accel_forearm_z"      "magnet_forearm_x"    
## [52] "magnet_forearm_y"     "magnet_forearm_z"
```

# Validation strategy

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
  
  
# Model Building
  
Following the authors of the original paper, we attempt prediction based on random forests.  Note that the formula interface for `randomForest()` is not used, as it generates high overhead when there are many variables, and that minimum `nodesize` is set to 10 (the default is 1).  These steps were taken to make run times feasible.  Below, the cumulative out-of-bag error is shown for every 20 trees, indicating that model performance is adequate by 500 (the default). 

```r
if (!"randomForest" %in% rownames(installed.packages())) install.packages("randomForest")
library(randomForest)
set.seed(921)
rfMod <- randomForest(x=Train[,-54], y=Train$classe, proximity = FALSE, nodesize = 10, do.trace = 20)
```

```
## ntree      OOB      1      2      3      4      5
##    20:   2.28%  0.97%  3.50%  3.28%  3.13%  1.34%
##    40:   1.26%  0.37%  2.04%  1.83%  1.95%  0.65%
##    60:   1.06%  0.29%  1.84%  1.53%  1.63%  0.45%
##    80:   0.97%  0.26%  1.65%  1.41%  1.68%  0.28%
##   100:   0.91%  0.21%  1.46%  1.41%  1.59%  0.36%
##   120:   0.81%  0.18%  1.23%  1.28%  1.50%  0.28%
##   140:   0.81%  0.13%  1.19%  1.28%  1.54%  0.36%
##   160:   0.78%  0.13%  1.19%  1.11%  1.68%  0.24%
##   180:   0.75%  0.13%  1.04%  1.02%  1.68%  0.32%
##   200:   0.73%  0.10%  0.96%  1.07%  1.68%  0.28%
##   220:   0.73%  0.13%  1.04%  1.11%  1.50%  0.28%
##   240:   0.75%  0.16%  1.00%  1.15%  1.59%  0.28%
##   260:   0.74%  0.08%  1.00%  1.19%  1.54%  0.32%
##   280:   0.72%  0.13%  1.00%  1.02%  1.59%  0.28%
##   300:   0.70%  0.13%  0.96%  1.07%  1.45%  0.28%
##   320:   0.72%  0.16%  1.00%  1.07%  1.50%  0.28%
##   340:   0.68%  0.16%  0.88%  0.98%  1.50%  0.28%
##   360:   0.70%  0.13%  0.92%  1.07%  1.50%  0.28%
##   380:   0.73%  0.13%  0.96%  1.15%  1.50%  0.32%
##   400:   0.71%  0.10%  0.92%  1.07%  1.50%  0.36%
##   420:   0.73%  0.10%  0.96%  1.15%  1.54%  0.32%
##   440:   0.71%  0.13%  0.92%  1.11%  1.50%  0.32%
##   460:   0.72%  0.13%  0.88%  1.15%  1.54%  0.32%
##   480:   0.71%  0.10%  0.92%  1.07%  1.45%  0.40%
##   500:   0.69%  0.08%  0.88%  1.07%  1.50%  0.36%
```

We next compute the confusion matrix and associated statistics on the validation set.  Performance appears quite good:


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

# Out-of-sample error

Based on the above, we expect an out-of-sample error rate of **0.94%**.  Note that this error rate is slightly higher than the 'out-of-bag' estimate of 0.69% from model training.  This is probably just a statistical fluctuation: the difference is inside the 95% confidence interval provided in the confusion matrix results.

# Conclusions

The method above provides high same-person recognition performance.  However, a training period is required on a person-by-person basis.  "Leave-one-participant-out" performance has not been evaluated here.

# Bibliography
  
  
