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
