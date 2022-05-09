install.packages("lime")
library(MASS)
library(lime)
data(biopsy)

# First we'll clean up the data a bit
biopsy$ID <- NULL
biopsy <- na.omit(biopsy)
names(biopsy) <- c('clump thickness', 'uniformity of cell size',
                   'uniformity of cell shape', 'marginal adhesion',
                   'single epithelial cell size', 'bare nuclei',
                   'bland chromatin', 'normal nucleoli', 'mitoses',
                   'class')

# Now we'll fit a linear discriminant model on all but 4 cases
set.seed(4)
test_set <- sample(seq_len(nrow(biopsy)), 1)
prediction <- biopsy$class
biopsy$class <- NULL
model <- lda(biopsy[-test_set, ], prediction[-test_set])
predict(model, biopsy[test_set, ])
explainer <- lime(biopsy[-test_set,], model, bin_continuous = TRUE, quantile_bins = FALSE)
explanation <- explain(biopsy[test_set, ], explainer, n_labels = 1, n_features = 4)
# Only showing part of output for better printing
explanation[, 2:9]
explanation <- explain(biopsy[test_set, ], explainer, n_labels = 1, n_features = 4, kernel_width = 0.5)
explanation[, 2:9]
plot_features(explanation, ncol = 1)



library(MASS)
iris_test <- iris[1, 1:4]
iris_train <- iris[-1, 1:4]
iris_lab <- iris[[5]][-1]
model <- lda(iris_train, iris_lab)
explanation <- lime(iris_train, model)
explanations <- explain(iris_test, explanation, n_labels = 1, n_features = 2)

# Get an overview with the standard plot
plot_features(explanations)

df <- read.csv("data/modelOut.csv", header = TRUE)
y <- ifelse(df$churn_Y==1,"1","0")
myvars <- c("dropperc", "income", "mins", "consecmonths")
df2 <- df[myvars]
#install.packages('xgboost')
library(caret)
model <- train(df2,y,method='xgbTree')
model2 <- lda(df2, y)
test_set <- sample(seq_len(nrow(df2)), 1)
test <- df2[test_set,]
explanation <- lime(df2, model2)
#explanation
explanations <- explain(test, explanation, n_labels = 1, n_features = 2)
plot_features(explanations)
plot_explanations(explanations)
