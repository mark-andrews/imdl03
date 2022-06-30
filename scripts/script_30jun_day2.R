library(tidyverse)
library(mlr)
library(mlbench)


# Decision trees ----------------------------------------------------------

data(Zoo)

# convert logical variables to factors
Zoo <- mutate(Zoo, across(where(is.logical), as.factor))

ZooTask <- makeClassifTask(data = Zoo, target = 'type')
dtree <- makeLearner('classif.rpart')
dtree_trained <- train(dtree, ZooTask)

library(rpart.plot)
dtree_M <- getLearnerModel(dtree_trained)
rpart.plot(dtree_M)

# evaluate performance
p3 <- predict(dtree_trained, newdata = Zoo)
calculateConfusionMatrix(p3)
performance(p3, measures = list(acc, mmce))

# Cross validation
kfold_cv <- makeResampleDesc(method = 'RepCV', folds = 10, reps = 10)
dtree_cv <- resample(dtree,
                     task = ZooTask,
                     resampling = kfold_cv,
                     measures = list(acc, mmce))
dtree_cv$aggr


# Optimize the hyper-params -----------------------------------------------

getParamSet(dtree)

dtree_param_space <- makeParamSet(
  makeIntegerParam('minsplit', lower = 3, upper = 25),
  makeIntegerParam('minbucket', lower = 3, upper = 25),
  makeNumericParam('cp', lower = 0.001, upper = 0.5),
  makeIntegerParam('maxdepth', lower = 3, upper = 20)
)

rand_search <- makeTuneControlRandom(maxit = 100)

kfold_cv <- makeResampleDesc(method = 'RepCV',
                             folds = 10,
                             reps = 10)

dtree_tune <- tuneParams(dtree,
                         task = ZooTask,
                         resampling = kfold_cv,
                         par.set = dtree_param_space,
                         control = rand_search)
dtree_tune$x

dtree2 <- setHyperPars(dtree, par.vals = dtree_tune$x)

# evaluate it
dtree2_cv <- resample(dtree2,
                      task = ZooTask,
                      resampling = kfold_cv,
                      measures = list(acc, mmce))
dtree2_cv$aggr


dtree2_trained <- train(dtree2, ZooTask)
dtree2_trained_m <- getLearnerModel(dtree2_trained)
rpart.plot(dtree2_trained_m)


# bootstrapping
x <- rnorm(10)
sample(x, size = length(x), replace = TRUE)



# Random forest -----------------------------------------------------------

rforest <- makeLearner('classif.randomForest')
rforest_trained_1 <- train(rforest, ZooTask)

p4 <- predict(rforest_trained_1, newdata = Zoo) # overfitting!
performance(p4, measures = list(acc, mmce))
calculateConfusionMatrix(p4)

# cross validate it
rforest_cv <- resample(rforest,
                       task = ZooTask,
                       resampling = kfold_cv,
                       measures = list(acc, mmce))

