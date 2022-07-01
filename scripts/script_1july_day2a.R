library(tidyverse)
library(mlr)


# Read in the data --------------------------------------------------------

blobs_df <- read_csv("https://raw.githubusercontent.com/mark-andrews/imdl03/main/data/blobs2.csv")

# plot it
ggplot(blobs_df,
       aes(x = x, y = y, colour = factor(label))) +
  geom_point()

ggplot(blobs_df,
       aes(x = x, y = y)) +
  geom_point()


# Create task and learner -------------------------------------------------

blobsTask <- makeClusterTask(
  data = as.data.frame(blobs_df %>% select(-label))
)

k_means <- makeLearner('cluster.kmeans',
                       iter.max = 50)

getParamSet(k_means)

k_means_par_set <- makeParamSet(
  makeDiscreteParam('centers', values = seq(8)),
  makeDiscreteParam('algorithm', values = c('Hartigan-Wong',
                                            'Lloyd',
                                            'MacQueen'))
)

grid1 <- makeTuneControlGrid()

kfold_cv <- makeResampleDesc('RepCV',
                             folds = 10,
                             reps = 10)

k_means_tuning <- tuneParams(learner = k_means,
                             task = blobsTask,
                             resampling = kfold_cv,
                             control = grid1,
                             par.set = k_means_par_set)

k_means_tuning$x

k_means_tuned <- setHyperPars(k_means, par.vals = k_means_tuning$x)

k_means_tuned_trained <- train(k_means_tuned, blobsTask)

k_means_model <- getLearnerModel(k_means_tuned_trained)
k_means_model$centers
k_means_model$cluster


blobs_df %>% 
  mutate(cluster = k_means_model$cluster) %>% 
  ggplot(aes(x = x, y = y, colour = factor(cluster))) +
  geom_point()

# Prob mixture models -----------------------------------------------------

library(mclust)

ggplot(faithful,
       aes(x = eruptions, y = waiting)
) + geom_point()


faithful_bic <- mclustBIC(faithful)
plot(faithful_bic)

mog <- Mclust(faithful, x = faithful_bic)
summary(mog)
plot(M, what = 'classification')
plot(M, what = 'density')
summary(mog, parameters = T)
