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

# Naive Bayes -------------------------------------------------------------

library(mlbench)
data("HouseVotes84")



# Set up naive Bayes ------------------------------------------------------

votesTask <- makeClassifTask(data = HouseVotes84, target = 'Class')
nbLearner <- makeLearner('classif.naiveBayes')
nbLearner_train <- train(nbLearner, votesTask)


# Evaluate ----------------------------------------------------------------

p <- predict(nbLearner_train, newdata = HouseVotes84)
calculateConfusionMatrix(p)
calculateROCMeasures(p)


# Cross validation --------------------------------------------------------

kfold_cv <- makeResampleDesc(method = 'RepCV',
                             folds = 10,
                             reps = 10,
                             stratify = T)

nbCV <- resample(learner = nbLearner,
                 task = votesTask,
                 resampling = kfold_cv,
                 measures = list(acc, mmce, tpr, ppv, f1))

nbCV$aggr


# Support vector machines -------------------------------------------------

library(kernlab)
data(spam)

table(spam$type)


# svm ---------------------------------------------------------------------

spamTask <- makeClassifTask(data = spam, target = 'type')
svmLearner <- makeLearner('classif.svm', kernel = 'linear')
svm_trained <- train(svmLearner, spamTask)

# performance -------------------------------------------------------------

p <- predict(svm_trained, newdata = spam)
calculateConfusionMatrix(p)
calculateROCMeasures(p)


kfold_cv <- makeResampleDesc(method = 'RepCV',
                             folds = 10,
                             reps = 3,
                             stratify = T)

spamCV <- resample(learner = svmLearner,
                 task = spamTask,
                 resampling = kfold_cv,
                 measures = list(acc, mmce, tpr, ppv, f1))

spamCV$aggr

# Optimize settings -------------------------------------------------------


getParamSet(svmLearner)

svm_param_space <- makeParamSet(
  makeNumericParam('cost', lower = 0.1, upper = 10),
  makeNumericParam('gamma', lower = 0.1, upper = 10)
)

randSearch <- makeTuneControlRandom(maxit = 25)

svm_tuned <- tuneParams('classif.svm',
                        task = spamTask,
                        resampling = makeResampleDesc('Holdout', split = 0.75),
                        par.set = svm_param_space,
                        control = randSearch)

svm_tuned$x

svmLearner_tuned <- setHyperPars(makeLearner('classif.svm'),
                                 par.vals = svm_tuned$x)

svmLearner_tuned_trained <- train(svmLearner_tuned, spamTask)

p1 <- predict(svmLearner_tuned_trained, newdata = spam)
calculateConfusionMatrix(p1)
calculateROCMeasures(p1)

# CV evaluation
spamCV2 <- resample(learner = svmLearner_tuned,
                   task = spamTask,
                   resampling = kfold_cv,
                   measures = list(acc, mmce, tpr, ppv, f1))
