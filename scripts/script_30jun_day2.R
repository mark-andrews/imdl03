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
