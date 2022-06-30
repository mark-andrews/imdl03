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

# Grid search -------------------------------------------------------------

space1 <- makeParamSet(
  makeDiscreteParam('minsplit', values = c(3, 5, 10, 15)),
  makeDiscreteParam('minbucket', values = c(5, 10, 15, 20)),
  makeDiscreteParam('cp', values = c(0.001, 0.01, 0.1, 0.5))
)

grid1 <- makeTuneControlGrid()

dtree_tune <- tuneParams(dtree,
                         task = ZooTask,
                         resampling = kfold_cv,
                         par.set = space1,
                         control = grid1)
dtree_tune$x


space2 <- makeParamSet(
  makeIntegerParam('minsplit', lower = 3, upper = 20),
  makeIntegerParam('minbucket', lower = 3, upper = 20),
  makeNumericParam('cp', lower = 0.001, upper = 0.5)
)

grid2 <- makeTuneControlGrid(resolution = 5)

dtree_tune <- tuneParams(dtree,
                         task = ZooTask,
                         resampling = kfold_cv,
                         par.set = space2,
                         control = grid2)
dtree_tune$x


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

rforest_cv$aggr

getParamSet(rforest)

rforest_param <- makeParamSet(
  makeIntegerParam('ntree', lower = 250, upper = 750),
  makeIntegerParam('nodesize', lower = 2, upper = 10),
  makeIntegerParam('maxnodes', lower = 5, upper = 10)
)

rand_search <- makeTuneControlRandom(maxit = 250)

rforest_tune <- tuneParams(rforest,
                           task = ZooTask,
                           resampling = kfold_cv,
                           par.set = rforest_param,
                           control = rand_search)

rforest_tune$x

rforest2 <- setHyperPars(rforest, par.vals = rforest_tune$x)

# cross validate it
rforest2_cv <- resample(rforest2,
                        task = ZooTask,
                        resampling = kfold_cv,
                        measures = list(acc, mmce))
rforest2_cv$aggr



# Load Keras library ------------------------------------------------------

library(keras)
mnist <- dataset_mnist()

dim(mnist$train$x)
dim(mnist$test$x)

mnist$train$y


x_train <- array_reshape(mnist$train$x, 
                         c(nrow(mnist$train$x), 28 ^ 2))
x_test <- array_reshape(mnist$test$x, 
                         c(nrow(mnist$test$x), 28 ^ 2))

x_train <- x_train /255
x_test <- x_test/255

y_train <- to_categorical(mnist$train$y, 10)
y_test <- to_categorical(mnist$test$y, 10)


# Define our model --------------------------------------------------------

nn_model <- keras_model_sequential()

nn_model %>% 
  layer_dense(input_shape = 28 ^ 2, units = 256, activation = 'relu') %>%
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 10, activation = 'softmax')

# these kinds of operations are happening in input layer to hidden layer
# W <- matrix(rnorm(784 * 256), nrow = 256, ncol = 784) # random weights
# W %*% input_1

summary(nn_model)

nn_model %>% 
  compile(loss = 'categorical_crossentropy',
          optimizer = optimizer_rmsprop(),
          metrics = 'accuracy')

nn_model %>% 
  fit(x_train,
      y_train,
      epochs = 25,
      batch_size = 100,
      validation_split = 0.2)

nn_model %>% keras::evaluate(x_test, y_test)
nn_model %>% predict(x_test) %>% head()

# Conv nets ---------------------------------------------------------------

cifar <- dataset_cifar10()
dim(cifar$train$x)
plot(cifar$train$x[1,,,] %>% as.raster(max = 255)) # frog
plot(cifar$train$x[2675,,,] %>% as.raster(max = 255)) # frog
plot(cifar$train$x[61,,,] %>% as.raster(max = 255)) # truck
plot(cifar$train$x[10101,,,] %>% as.raster(max = 255)) #car


convnet <- keras_model_sequential()
convnet %>% 
  layer_conv_2d(input_shape = c(32, 32, 3),
                activation = 'relu',
                filters = 32, 
                kernel_size = c(3, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 64,
                kernel_size = c(3, 3),
                activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_flatten() %>% 
  layer_dense(units = 10, activation = 'softmax')
  
  
  