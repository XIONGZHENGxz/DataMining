#install.packages("glmnet")
library(glmnet)
#install.packages("cvTools")
library(cvTools)

## import the diabetes dataset
diabetes <- read.table("/Users/Amaranth/Downloads/HW3/Q1/diabetes.txt", sep=" ", col.names=c("age", "sex", "bmi", "map", "tc", "ldl", "hdl", "tch", "ltg", "glu", "y"), header=TRUE)
X <- diabetes[, 1:10]
y <- diabetes[, 11]

## add additional interaction variables to the features matrix
new_X <- poly(as.matrix(X), degree=2, raw=TRUE)

## set the seed to make your partition reproductible
training_size <- floor(0.78 * nrow(new_X))

set.seed(123)
train_idx <- sample(seq_len(nrow(new_X)), size=training_size)

yy <- array(y, c(nrow(new_X),1))

train.x <- new_X[train_idx, ]
train.y <- array(yy[train_idx,],c(training_size,1))
test.x <- new_X[-train_idx, ]
test.y <- array(yy[-train_idx,],c(nrow(new_X)-training_size,1))

library(ggplot2)
p <- ggplot(pairs(X))

alphas <- 10^seq(from = -2, to = 10, length.out = 100)

ridge.model <- glmnet(train.x, train.y, lambda=alphas, alpha = 0);
ridge.cv <- cv.glmnet(train.x, train.y, lambda=alphas, alpha = 0, nfold=5);

best_ridge <- ridge.cv$lambda.min
cat("Best chosen lambda for ridge is ", best_ridge)

lasso.model <- glmnet(train.x, train.y, lambda=alphas, alpha = 1);
lasso.cv <- cv.glmnet(train.x, train.y, lambda=alphas, alpha = 1, nfold=5);

best_lasso <- lasso.cv$lambda.min
cat("Best chosen lambda for lasso is " ,best_lasso)

# C
plot(ridge.model, xvar="lambda")   #  coefficient paths of ridge
plot(lasso.model, xvar="lambda")   #  coefficient paths of lasso

# D

predicted.lasso.value <- predict(lasso.cv, newx = test.x, s = "lambda.min");
cat("MSE for lasso is ", mean((predicted.lasso.value - test.y)^2))

predicted.ridge.value <- predict(ridge.cv, newx = test.x, s = "lambda.min");
cat("MSE for ridge is ",mean((predicted.ridge.value - test.y)^2))

