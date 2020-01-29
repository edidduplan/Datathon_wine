#============ 1) Import libraries ===================
pacman::p_load(readr, tidyverse, plotly)
pacman::p_load(rstudioapi)
pacman::p_load(caret, caretEnsemble, corrplot, data.table)
pacman::p_load(kknn, e1071, caTools)
pacman::p_load(DMwR, ROSE)

#============ Modelling with down/up-sampling ==================
set.seed(5)
trainsize <- createDataPartition(y = df_clean4$quality, 
                                 p = .75, list = F)
trainset <- df[trainsize,]
testset <- df[-trainsize,]

dim(filter(testset, quality == 9))

#Distribution of the train and testsets
ggplot(trainset, aes(quality)) +
  geom_histogram(stat = "count")

ggplot(testset, aes(quality)) +
  geom_histogram(stat = "count")

#Modelling
fit <- c()
prediction <- c()
performance <- c()
error_test_compare  <- c()

models <- c("svmRadial", "svmPoly","kknn")

ctrl <- trainControl(
  method = "repeatedcv", 
  number = 5, 
  repeats = 1,
  sampling = "up")

time_start <- Sys.time()
for (i in models){
  fit <- train(quality ~ . , 
               data = trainset, 
               method = i,
               tuneLength = 3,
               trControl = ctrl,
               preProc = c("center", "scale"))
  prediction_test <- predict(fit, newdata = testset)
  
  performance_test <- postResample(prediction_test, 
                                   testset$quality)
  
  error_test_compare <- cbind(error_test_compare, performance_test)
}
time_end <- Sys.time()

colnames(error_test_compare) <- c("svmRadial", "svmPoly","kknn")
error_dfclean4_up <- error_test_compare

#===== Further investigating with RF and C50 =======
set.seed(5)
trainsize <- createDataPartition(y = df_clean3$quality, 
                                 p = .75, list = F)
trainset <- df_clean3[trainsize,]
testset <- df_clean3[-trainsize,]

dim(filter(testset, quality == 9))

#Distribution of the train and testsets
ggplot(trainset, aes(quality)) +
  geom_histogram(stat = "count")

ggplot(testset, aes(quality)) +
  geom_histogram(stat = "count")

#Modelling 
trainset_smote <- SMOTE(quality ~ ., trainset, perc.over = 5000,perc.under=4900)
trainset_smote <- SMOTE(quality ~ ., trainset_smote, perc.over = 5000,perc.under=4900)
table(trainset$quality)
table(trainset_smote$quality)

fun_sample <- function(df, n){
  df[sample(1:nrow(df), n, replace = F),]
}

trainset_smote <- fun_sample(trainset_smote, 1000)

any(is.na(trainset_smote))

trainset_smote <- trainset_smote[complete.cases(trainset_smote), ]

ctrl <- trainControl(
  method = "repeatedcv", 
  number = 5, 
  repeats = 1,
  sampling = "up")

time_start <- Sys.time()
fit <- train(quality ~ . , 
             data = trainset, 
             method = "rf",
             tuneLength = 3,
             trControl = ctrl,
             preProc = c("center", "scale"))
prediction_test <- predict(fit, newdata = testset)
performance_rf_clean3_up <- postResample(prediction_test, testset$quality)
time_end <- Sys.time()

#New approach: downsampling first and then upsampling
#Downsampling
trainset$quality <- as.integer(trainset$quality)
trainset <- filter(trainset, (quality == 3 | quality == 4 | quality == 5))
trainset$quality <- as.factor(trainset$quality)
levels(trainset$quality) <- c("5", "6", "7")
down_train <- downSample(x = trainset[, -ncol(trainset)],
                         y = trainset$quality,
                         yname = "quality")

ggplot(down_train, aes(quality)) +
  geom_histogram(stat = "count")

extra <- filter(trainset, (quality == 3 | quality == 4 | quality == 8 | quality == 9))

down_train <- rbind(down_train, extra)

#Modelling
ctrl <- trainControl(
  method = "repeatedcv", 
  number = 5, 
  repeats = 1,
  sampling = "up")

time_start <- Sys.time()
fit <- train(quality ~ . , 
             data = down_train, 
             method = "rf",
             tuneLength = 3,
             trControl = ctrl,
             preProc = c("center", "scale"))
prediction_test <- predict(fit, newdata = testset)
performance_rf_clean3_dup <- postResample(prediction_test, testset$quality)
time_end <- Sys.time()