
plot_mnist <- function(data_df, rm_label = F){
  plt <- data_df %>% pivot_longer(cols = starts_with('px__'),
                                  names_to = c('x', 'y'),
                                  names_pattern = 'px__([0-9]*)_([0-9]*)',
                                  values_to = 'value') %>% 
    mutate(across(c(x,y), as.numeric)) %>% 
    ggplot(aes(x, y, fill = value)) +
    geom_tile() +
    facet_wrap(~ instance + label) +
    scale_fill_gradient(low = 'black', high = 'white') 
  
  if (rm_label){
    plt + theme(
      strip.background = element_blank(),
      strip.text.x = element_blank()
    )
  } else {
    plt
  }
  
}




# Subsample the data frame to the 3 and 7 ---------------------------------

mnist_df2 <- mnist_df %>%
  filter(label %in% c(3, 7))

X <- mnist_df2 %>% select(starts_with('px__')) %>% as.matrix()
nzv <- caret::nearZeroVar(X)
image(matrix(1:(28^2) %in% nzv, 28, 28))

col_index <- setdiff(1:ncol(X), nzv)

X <- X[, col_index]
mu <- mean(X)
s <- sd(as.vector(X))
data_df <- as.data.frame((X - mu)/s) %>% 
  mutate(target = if_else(mnist_df2$label == 7, '7', '3'))

# Test/train split --------------------------------------------------------

train_idx <- sample(1:nrow(data_df), 0.8 * nrow(data_df))
test_idx <- setdiff(1:nrow(data_df), train_idx)

train_df <- data_df[train_idx,]
test_df <- data_df[test_idx,]




