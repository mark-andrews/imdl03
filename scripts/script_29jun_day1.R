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
