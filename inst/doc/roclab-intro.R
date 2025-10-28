## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup--------------------------------------------------------------------
library(roclab)

## -----------------------------------------------------------------------------
set.seed(123)
n_lin <- 1500
n_pos_lin <- round(0.2 * n_lin)
n_neg_lin <- n_lin - n_pos_lin

X_train_lin <- rbind(
  matrix(rnorm(2 * n_neg_lin, mean = -1), ncol = 2),
  matrix(rnorm(2 * n_pos_lin, mean =  1), ncol = 2)
)
y_train_lin <- c(rep(-1, n_neg_lin), rep(1, n_pos_lin))

# Fit a linear model
fit_lin <- roclearn(X_train_lin, y_train_lin, lambda = 0.1)

# Summary
summary(fit_lin)

n_test_lin <- 300
n_pos_test_lin <- round(0.2 * n_test_lin)
n_neg_test_lin <- n_test_lin - n_pos_test_lin
X_test_lin <- rbind(
  matrix(rnorm(2 * n_neg_test_lin, mean = -1), ncol = 2),
  matrix(rnorm(2 * n_pos_test_lin, mean =  1), ncol = 2)
)
y_test_lin <- c(rep(-1, n_neg_test_lin), rep(1, n_pos_test_lin))

# Predict decision scores
pred_score_lin <- predict(fit_lin, X_test_lin, type = "response")
head(pred_score_lin)

# Predict classes {-1, 1}
pred_class_lin <- predict(fit_lin, X_test_lin, type = "class")
head(pred_class_lin)

# AUC on the test set
auc(fit_lin, X_test_lin, y_test_lin)

## -----------------------------------------------------------------------------
set.seed(123)
n_ker <- 1500
r_train_ker <- sqrt(runif(n_ker, 0.05, 1))
theta_train_ker <- runif(n_ker, 0, 2*pi)
X_train_ker <- cbind(r_train_ker * cos(theta_train_ker), r_train_ker * sin(theta_train_ker))
y_train_ker <- ifelse(r_train_ker < 0.5, 1, -1)

# Fit a kernel model
fit_ker <- kroclearn(X_train_ker, y_train_ker, lambda = 0.1, kernel = "radial")

# Summary
summary(fit_ker)

n_test_ker <- 300
r_test_ker <- sqrt(runif(n_test_ker, 0.05, 1))
theta_test_ker <- runif(n_test_ker, 0, 2*pi)
X_test_ker <- cbind(r_test_ker * cos(theta_test_ker), r_test_ker * sin(theta_test_ker))
y_test_ker <- ifelse(r_test_ker < 0.5, 1, -1)

# Predict decision scores
pred_score_ker <- predict(fit_ker, X_test_ker, type = "response")
head(pred_score_ker)

# Predict classes {-1, 1}
pred_class_ker <- predict(fit_ker, X_test_ker, type = "class")
head(pred_class_ker)

# AUC on the test set
auc(fit_ker, X_test_ker, y_test_ker)

## -----------------------------------------------------------------------------
# 5-fold CV for linear models
cvfit_lin <- cv.roclearn(
  X_train_lin, y_train_lin,
  lambda.vec = exp(seq(log(0.01), log(5), length.out = 20)),
  nfolds = 5
)

# Summarize the cross-validation result
summary(cvfit_lin)

## ----fig.width=7, fig.height=6------------------------------------------------
# Plot the cross-validation AUC across lambda values
plot(cvfit_lin)

## -----------------------------------------------------------------------------
# 5-fold CV for kernel models
cvfit_ker <- cv.kroclearn(
  X_train_ker, y_train_ker,
  lambda.vec = exp(seq(log(0.01), log(5), length.out = 20)),
  kernel = "radial",
  nfolds = 5
)

# Summarize the cross-validation result
summary(cvfit_ker)

## ----fig.width=7, fig.height=6------------------------------------------------
# Plot the cross-validation AUC across lambda values
plot(cvfit_ker)

## -----------------------------------------------------------------------------
library(mlbench)
data(Ionosphere)

# Prepare data
X_iono <- Ionosphere[, -35]
y_iono <- ifelse(Ionosphere$Class == "bad", 1, -1)

set.seed(123)
train_idx <- sample(seq_len(nrow(X_iono)), size = 200)
X_train_iono <- X_iono[train_idx, ]
y_train_iono <- y_iono[train_idx]
X_test_iono  <- X_iono[-train_idx, ]
y_test_iono  <- y_iono[-train_idx]

# Fit a linear model
fit_iono_lin <- roclearn(X_train_iono, y_train_iono, lambda = 0.1, approx=TRUE)
summary(fit_iono_lin)

# Predict decision scores
pred_score_iono_lin <- predict(fit_iono_lin, X_test_iono, type = "response")
head(pred_score_iono_lin)

# Predict classes {-1, 1}
pred_class_iono_lin <- predict(fit_iono_lin, X_test_iono, type = "class")
head(pred_class_iono_lin)

# AUC on the test set
auc(fit_iono_lin, X_test_iono, y_test_iono)

## -----------------------------------------------------------------------------
# 5-fold CV for linear models
cvfit_iono_lin <- cv.roclearn(
  X_train_iono, y_train_iono,
  lambda.vec = exp(seq(log(0.01), log(5), length.out = 10)),
  approx=TRUE, nfolds=5)
summary(cvfit_iono_lin)

## ----fig.width=7, fig.height=6------------------------------------------------
# Plot the cross-validation AUC across lambda values
plot(cvfit_iono_lin)

## -----------------------------------------------------------------------------
# Fit a kernel model
fit_iono_ker <- kroclearn(X_train_iono, y_train_iono, lambda = 0.1, kernel = "radial", approx=TRUE)
summary(fit_iono_ker)

# Predict decision scores
pred_score_iono_ker <- predict(fit_iono_ker, X_test_iono, type = "response")
head(pred_score_iono_ker)

# Predict classes {-1, 1}
pred_class_iono_ker <- predict(fit_iono_ker, X_test_iono, type = "class")
head(pred_class_iono_ker)

# AUC on the test set
auc(fit_iono_ker, X_test_iono, y_test_iono)

## -----------------------------------------------------------------------------
# 5-fold CV for kernel models
cvfit_iono_ker <- cv.kroclearn(
  X_train_iono, y_train_iono,
  lambda.vec = exp(seq(log(0.01), log(5), length.out = 10)),
  kernel = "radial",
  approx=TRUE, nfolds=5)
summary(cvfit_iono_ker)

## ----fig.width=7, fig.height=6------------------------------------------------
# Plot the cross-validation AUC across lambda values
plot(cvfit_iono_ker)

## -----------------------------------------------------------------------------
library(kernlab)
data(spam)

# Prepare data
X_spam <- spam[, -58]
y_spam <- ifelse(spam$type == "spam", 1, -1)

set.seed(123)
train_idx <- sample(seq_len(nrow(X_spam)), size = 3000)
X_train_spam <- X_spam[train_idx, ]
y_train_spam <- y_spam[train_idx]
X_test_spam  <- X_spam[-train_idx, ]
y_test_spam  <- y_spam[-train_idx]

# Fit a linear model
fit_spam_lin <- roclearn(X_train_spam, y_train_spam, lambda = 0.1)
summary(fit_spam_lin)

# Predict decision scores
pred_score_spam_lin <- predict(fit_spam_lin, X_test_spam, type = "response")
head(pred_score_spam_lin)

# Predict classes {-1, 1}
pred_class_spam_lin <- predict(fit_spam_lin, X_test_spam, type = "class")
head(pred_class_spam_lin)

# AUC on the test set
auc(fit_spam_lin, X_test_spam, y_test_spam)

## -----------------------------------------------------------------------------
# 5-fold CV for linear models 
cvfit_spam_lin <- cv.roclearn(
  X_train_spam, y_train_spam,
  lambda.vec = exp(seq(log(0.01), log(5), length.out = 10)), nfolds=5)
summary(cvfit_spam_lin)

## ----fig.width=7, fig.height=6------------------------------------------------
# Plot the cross-validation AUC across lambda values
plot(cvfit_spam_lin)

## -----------------------------------------------------------------------------
# Fit a kernel model
fit_spam_ker <- kroclearn(X_train_spam, y_train_spam, lambda = 0.1, kernel = "radial")
summary(fit_spam_ker)

# Predict decision scores
pred_score_spam_ker <- predict(fit_spam_ker, X_test_spam, type = "response")
head(pred_score_spam_ker)

# Predict classes {-1, 1}
pred_class_spam_ker <- predict(fit_spam_ker, X_test_spam, type = "class")
head(pred_class_spam_ker)

# AUC on the test set
auc(fit_spam_ker, X_test_spam, y_test_spam)

## -----------------------------------------------------------------------------
# 5-fold CV for kernel models 
cvfit_spam_ker <- cv.kroclearn(
  X_train_spam, y_train_spam,
  kernel = "radial", 
  lambda.vec = exp(seq(log(0.01), log(5), length.out = 10)), nfolds=5)
summary(cvfit_spam_ker)

## ----fig.width=7, fig.height=6------------------------------------------------
# Plot the cross-validation AUC across lambda values
plot(cvfit_spam_ker)

