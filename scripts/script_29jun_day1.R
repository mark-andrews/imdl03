library(mlr)
library(tidyverse)

# Read in data ------------------------------------------------------------

affairs_df <- read_csv("https://raw.githubusercontent.com/mark-andrews/imdl03/main/data/affairs.csv")

training_df <- affairs_df %>% 
  # create binary target variable
  mutate(affair = if_else(affairs > 0, 'yes', 'no')) %>%
  # remove the `affairs` variable
  select(-affairs) %>% 
  # convert character vectors to factors
  mutate(across(where(is.character), as.factor)) %>%
  # relevel the target factor
  mutate(affair = factor(affair, levels = c('yes', 'no'))) %>% 
  # convert to base R data frame
  as.data.frame()
  

# Make "task" and "learner" -----------------------------------------------

affairsTask <- makeClassifTask(data = training_df,
                               target = 'affair')

logReg <- makeLearner("classif.logreg", predict.type = 'prob')


# train it
logReg_trained <- train(logReg, affairsTask)


# Evaluate performance ----------------------------------------------------

p <- predict(logReg_trained, newdata = training_df)
calculateConfusionMatrix(p)
calculateROCMeasures(p)

# recall, aka TPR aka sensivity
recall <- measureTPR(p$data$truth, p$data$response, positive = 'yes')
# precision, ppv
precision <- measurePPV(p$data$truth, p$data$response, positive = 'yes')

# F1
1/mean(1/c(recall, precision))
measureF1(p$data$truth, p$data$response, positive = 'yes')

# Cross validation --------------------------------------------------------

kfold_cv <- makeResampleDesc(method = 'RepCV',
                             folds = 10,
                             reps = 10,
                             stratify = TRUE)

logRegKfold <- resample(logReg,
                        affairsTask,
                        resampling = kfold_cv,
                        measures = list(acc, mmce, tpr, ppv, f1))

logRegKfold$aggr


# ROC Curve ---------------------------------------------------------------

roc_df <- generateThreshVsPerfData(p, 
                                   measures = list(fpr, tpr))
plotROCCurves(roc_df)
performance(p, measures = auc)


# Get the glm model -------------------------------------------------------

logReg_trained_M <- getLearnerModel(logReg_trained)


# Get mnist data ----------------------------------------------------------

mnist_df <- readRDS('data/mnist.Rds')

# Read in some utils etc --------------------------------------------------

source("https://raw.githubusercontent.com/mark-andrews/imdl03/main/scripts/mnist_utils.R")


# Plot sample -------------------------------------------------------------

plot_mnist(mnist_df %>% sample_n(16))
plot_mnist(mnist_df %>% sample_n(16), rm_label = T)


# Make task, learner and then train it ------------------------------------

mnistTask <- makeClassifTask(data = train_df, 
                             target = 'target')
mnistLogReg <- makeLearner('classif.logreg', predict.type = 'prob')
mnistLogReg_trained <- train(mnistLogReg, mnistTask)


# Evaluate performance ----------------------------------------------------

p <- predict(mnistLogReg_trained, newdata = test_df)
calculateConfusionMatrix(p)
calculateROCMeasures(p)


# Cross validation --------------------------------------------------------

kfold_cv <- makeResampleDesc(method = 'CV', iters = 10, stratify = TRUE)

mnistKfold <- resample(mnistLogReg,
                       mnistTask,
                       resampling = kfold_cv,
                       measures = list(acc, mmce, ppv, tpr, f1, auc))

mnistKfold$aggr
