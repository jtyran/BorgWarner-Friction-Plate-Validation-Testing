library(dplyr)
library(fpp2)
library(lubridate)
library(readxl)
library(tidyverse)
library(readxl)
library(ggplot2)
library(ggfortify)
library(zoo)
library(forecast)
library(cluster)
library(dplyr)
library(tidyverse)
library(factoextra)

# Read data
valid <- valid_raw[,-1]
train <- train_raw[,-1]

str(train)
str(valid)

names(train)[10] <- "TestResult"
names(valid)[10] <- "TestResult"
names(train)[1:9] <- paste0("var", 1:9)
names(valid)[1:9] <- paste0("var", 1:9)

names(train)

# Convert TRUE/FALSE -> factor(Pass/Fail)
train$TestResult <- factor(train$TestResult, levels = c("FALSE", "TRUE"))
valid$TestResult <- factor(valid$TestResult, levels = c("FALSE", "TRUE"))

# Logistic Regression Model
log_model <- glm(
  TestResult ~ var1 + var2 + var3 + var4 + var5 + var6 + var7 + var8 + var9,
  data = train,
  family = binomial
)

summary(log_model)

# Predictions
pred_prob <- predict(log_model, valid, type = "response")
pred_class <- ifelse(pred_prob > 0.5, "TRUE", "FALSE")
pred_class <- factor(pred_class, levels = c("FALSE", "TRUE"))

# Confusion Matrix
cm <- table(Predicted = pred_class, Actual = valid$TestResult)
cm

# Accuracy
accuracy <- mean(pred_class == valid$TestResult)
accuracy

prop.table(table(train$TestResult))


library(rpart)
library(rpart.plot)
install.packages("gbm")
library(randomForest)    
library(gbm)

tree_model <- rpart(
  TestResult ~ var1 + var2 + var3 + var4 + var5 + var6 + var7 + var8 + var9,
  data = train,
  method = "class",
  control = rpart.control(cp = 0.001)  # allows deeper trees
)

rpart.plot(tree_model)


tree_pred <- predict(tree_model, valid, type = "class")
table(Predicted = tree_pred, Actual = valid$TestResult)

mean(tree_pred == valid$TestResult)   # accuracy

# Random Forest Model
set.seed(123)

rf_model <- randomForest(
  TestResult ~ var1 + var2 + var3 + var4 + var5 + var6 + var7 + var8 + var9,
  data = train,
  ntree = 500,
  mtry = 3,
  importance = TRUE
)

rf_model

rf_pred <- predict(rf_model, valid, type = "class")
table(Predicted = rf_pred, Actual = valid$TestResult)
mean(rf_pred == valid$TestResult)
varImpPlot(rf_model)

# Boosted Trees -gbm-
set.seed(123)

boost_model <- gbm(
  formula = as.numeric(TestResult) - 1 ~ var1 + var2 + var3 + var4 + var5 + var6 + var7 + var8 + var9,
  data = train,
  distribution = "bernoulli",
  n.trees = 3000,
  interaction.depth = 3,
  shrinkage = 0.01,
  n.minobsinnode = 10,
  verbose = FALSE
)

boost_prob <- predict(boost_model, valid, n.trees = 3000, type = "response")
boost_pred <- ifelse(boost_prob > 0.5, "TRUE", "FALSE")
boost_pred <- factor(boost_pred, levels = levels(valid$TestResult))

table(Predicted = boost_pred, Actual = valid$TestResult)
mean(boost_pred == valid$TestResult)

acc_log   <- mean(pred_class == valid$TestResult)
acc_tree  <- mean(tree_pred == valid$TestResult)
acc_rf    <- mean(rf_pred == valid$TestResult)
acc_boost <- mean(boost_pred == valid$TestResult)

model_compare <- data.frame(
  Model = c("Logistic Regression", "Decision Tree", "Random Forest", "Boosted Tree"),
  Accuracy = c(acc_log, acc_tree, acc_rf, acc_boost)
)

model_compare

library(caret)

eval_metrics <- function(pred, actual) {
  cm <- confusionMatrix(pred, actual, positive = "TRUE")
  list(
    Accuracy  = cm$overall["Accuracy"],
    Precision = cm$byClass["Precision"],
    Recall    = cm$byClass["Recall"],
    F1        = cm$byClass["F1"]
  )
}

eval_metrics(boost_pred, valid$TestResult)

library(nnet)

set.seed(123)

nn_model <- nnet(
  TestResult ~ var1 + var2 + var3 + var4 + var5 + var6 + var7 + var8 + var9,
  data = train,
  size = 5,
  maxit = 500
)

nn_prob <- predict(nn_model, valid, type = "raw")
nn_pred <- ifelse(nn_prob > 0.5, "TRUE", "FALSE")
nn_pred <- factor(nn_pred, levels = levels(valid$TestResult))

mean(nn_pred == valid$TestResult)

model_compare <- rbind(
  model_compare,
  data.frame(Model="Neural Network", Accuracy = mean(nn_pred == valid$TestResult))
)

model_compare

# K-Means Cluster Analysis
cluster_data <- train[,1:9]
set.seed(123)
km <- kmeans(cluster_data, centers = 2, nstart = 25)
km$cluster

table(Cluster = km$cluster, Actual = train$TestResult)


